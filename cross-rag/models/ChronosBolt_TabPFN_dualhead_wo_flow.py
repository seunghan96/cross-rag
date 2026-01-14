import copy
import logging
import warnings
from dataclasses import dataclass, fields
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

from .base import BaseChronosPipeline, ForecastType

logger = logging.getLogger("autogluon.timeseries.models.chronos")


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length % self.patch_size != 0:
            padding_size = (*x.shape[:-1], self.patch_size - (length % self.patch_size))
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.concat((padding, x), dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else:
            loc, scale = loc_scale

        normalized = (x - loc) / scale
        is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True) if loc_scale is None else (scale == 1)
        normalized = torch.where(is_constant, torch.ones_like(normalized), normalized)
        return normalized, (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loc, scale = loc_scale
        is_constant = scale == 1
        return torch.where(is_constant, loc, x * scale + loc)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res
        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        super().__init__(config)
        self.model_dim = config.d_model
        config_fields = {f.name for f in fields(ChronosBoltConfig)}
        self.chronos_config = ChronosBoltConfig(**{k: v for k, v in config.chronos_config.items() if k in config_fields})
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1
        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )
        self.instance_norm = InstanceNorm()
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self._init_decoder(config)
        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )
        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()
            module.residual_layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()
            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)
        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0
        input_embeds = self.input_patch_embedding(patched_context)
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)
        encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
        hidden_states = encoder_outputs[0]
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)
        quantile_preds_shape = (batch_size, self.num_quantiles, self.chronos_config.prediction_length)
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)
        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            assert self.chronos_config.prediction_length >= target.shape[-1]
            target = target.to(quantile_preds.device)
            target_mask = target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            target[~target_mask] = 0.0
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)
            loss = (
                2
                * torch.abs((target - quantile_preds) * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1)))
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)
            loss = loss.sum(dim=-1)
            loss = loss.mean()
        quantile_preds = self.instance_norm.inverse(quantile_preds.view(batch_size, -1), loc_scale).view(*quantile_preds_shape)
        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(self, input_embeds, attention_mask, hidden_states, output_attentions=False):
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.config.decoder_start_token_id, device=input_embeds.device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return decoder_outputs.last_hidden_state


class ChronosBoltPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048
    _aliases = ["PatchedT5Pipeline"]

    def __init__(self, model: ChronosBoltModelForForecasting):
        super().__init__(inner_model=model)
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ):
        context_tensor = self._prepare_and_validate_context(context=context)
        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length
        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)
        predictions = []
        remaining = prediction_length
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor.to(device=self.model.device, dtype=torch.float32),
                ).quantile_preds.to(context_tensor)
            predictions.append(prediction)
            remaining -= prediction.shape[-1]
            if remaining <= 0:
                break
            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]
            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)
        return torch.cat(predictions, dim=-1)[..., :prediction_length]

    def predict_quantiles(
        self, context: torch.Tensor, prediction_length: int, quantile_levels: List[float], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = (
            self.predict(
                context,
                prediction_length=prediction_length,
            )
            .detach()
            .cpu()
            .swapaxes(1, 2)
        )
        training_quantile_levels = self.quantiles
        if set(quantile_levels).issubset(set(training_quantile_levels)):
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(training_quantile_levels):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions, q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype), dim=-1
            ).permute(1, 2, 0)
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.pop("optimization_strategy", None)
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        context_length = kwargs.pop("context_length", None)
        if context_length is not None:
            config.chronos_config["context_length"] = context_length
        architecture = config.architectures[0]
        class_ = globals().get(architecture)
        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting")
            class_ = ChronosBoltModelForForecasting
        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)


def compute_time_series_stats(tensor, dim=-1, keepdim=False):
    mean_val = tensor.mean(dim=dim, keepdim=keepdim)
    std_val = tensor.std(dim=dim, keepdim=keepdim)
    min_val = tensor.min(dim=dim, keepdim=keepdim)[0]
    max_val = tensor.max(dim=dim, keepdim=keepdim)[0]
    return torch.cat([mean_val, std_val, min_val, max_val], dim=-1)


class ChronosBoltModelForForecastingWithRetrieval(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, augment: str):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        super().__init__(config)
        self.model_dim = config.d_model
        self.augment = augment
        self.eps_gate = 1e-4
        self.eps_softmax = 1e-4
        self.clamp_max = 1e4
        self.clamp_min = -1e4

        config_fields = {f.name for f in fields(ChronosBoltConfig)}
        self.chronos_config = ChronosBoltConfig(**{k: v for k, v in config.chronos_config.items() if k in config_fields})
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1
        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )
        self.instance_norm = InstanceNorm()
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self._init_decoder(config)
        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )
        self.dropout = nn.Dropout(p=0.2)

        if 'gate' in self.augment:
            self.gate_layer = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length * 2 + 512 + 12, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.gate_linear1 = nn.Linear(self.chronos_config.prediction_length * 2 + 512 + 12, config.d_model)
            self.gate_linear2 = nn.Linear(config.d_model, 1)

        if 'moe' in self.augment:
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.cross_mha = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=8, batch_first=True)
            self.self_mha = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=8, batch_first=True)
            self.ffn_cross = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ffn_self = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.mix_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 1),
            )
            self.mix_lambda_init = float(os.getenv("TABPFN_DUAL_LAMBDA_INIT", "0.5"))

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()
            module.residual_layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()
            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()

    def init_extra_weights(self, layers):
        factor = self.config.initializer_factor
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                self.init_extra_weights(layer)
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
                if layer.bias is not None:
                    layer.bias.data.zero_()
            if isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                if layer.in_proj_bias is not None:
                    layer.in_proj_bias.data.zero_()
                if layer.out_proj.bias is not None:
                    layer.out_proj.bias.data.zero_()

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)
        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        context, loc_scale = self.instance_norm(context)

        # numeric guards on retrieval inputs
        retrieved_seq = torch.nan_to_num(retrieved_seq, nan=0.0, posinf=0.0, neginf=0.0)
        distances = torch.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)

        if 'moe' not in self.augment:
            distances_clamped = distances.clamp(self.clamp_min, self.clamp_max)
            weights = torch.softmax(-distances_clamped, dim=1)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            retrieved_seq = (weights.unsqueeze(-1) * retrieved_seq).sum(dim=1)
            retrieved_seq = retrieved_seq.unsqueeze(1)

        L = self.chronos_config.prediction_length
        r_B, r_M, r_L = retrieved_seq.shape
        assert r_L % 2 == 0, "L of retrieved_seq should be even"
        retrieved_x, retrieved_y = retrieved_seq.split((r_L - L, L), dim=2)

        context = context.to(self.dtype)
        mask = mask.to(self.dtype)
        retrieved_seq = retrieved_seq.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0
        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
        hidden_states = encoder_outputs[0]
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        if self.augment == 'moe':
            retrieved_y_enc = []
            for i in range(r_M):
                retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
            retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)
            retrieved_y_enc = torch.nan_to_num(retrieved_y_enc, nan=0.0, posinf=0.0, neginf=0.0)

            cross_out, _ = self.cross_mha(sequence_output, retrieved_y_enc, retrieved_y_enc)
            cross_out = sequence_output + cross_out
            cross_out = cross_out + self.dropout(self.ffn_cross(cross_out))

            self_out, _ = self.self_mha(retrieved_y_enc, retrieved_y_enc, retrieved_y_enc)
            self_out = retrieved_y_enc + self_out
            self_out = self_out + self.dropout(self.ffn_self(self_out))
            self_pool = self_out.mean(dim=1, keepdim=True)

            mix_in = torch.cat([cross_out, self_pool], dim=-1)
            gate = torch.sigmoid(self.mix_gate(mix_in))
            gate = torch.clamp(gate, self.eps_gate, 1.0 - self.eps_gate)
            fused = gate * cross_out + (1 - gate) * self_pool

            sequence_output = fused

        quantile_preds_shape = (batch_size, self.num_quantiles, self.chronos_config.prediction_length)
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        fused_quantile_preds = quantile_preds

        if self.augment == 'gate':
            retrieved_y_rep = retrieved_y.repeat(1, self.num_quantiles, 1)
            x = context.unsqueeze(1).repeat(1, self.num_quantiles, 1)
            stats_preds = compute_time_series_stats(quantile_preds, dim=2, keepdim=True)
            stats_retrieved = compute_time_series_stats(retrieved_y_rep, dim=2, keepdim=True)
            stats_x = compute_time_series_stats(x, dim=2, keepdim=True)
            concat_preds = torch.cat([quantile_preds, retrieved_y_rep, x, stats_preds, stats_retrieved, stats_x], dim=-1).view(batch_size, self.num_quantiles, -1)
            gate = self.gate_layer(concat_preds) + self.gate_linear1(concat_preds)
            gate = torch.sigmoid(self.gate_linear2(gate))
            gate = torch.clamp(gate, self.eps_gate, 1.0 - self.eps_gate)
            fused_quantile_preds = gate * quantile_preds + (1 - gate) * retrieved_y_rep

        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(fused_quantile_preds.device)
            target_mask = target_mask.unsqueeze(1).to(fused_quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            target[~target_mask] = 0.0

            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (target - fused_quantile_preds)
                    * ((target <= fused_quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)
            loss = loss.sum(dim=-1)
            loss = loss.mean()

        fused_quantile_preds = torch.nan_to_num(fused_quantile_preds, nan=0.0, posinf=0.0, neginf=0.0)
        fused_quantile_preds = self.instance_norm.inverse(
            fused_quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=fused_quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(self, input_embeds, attention_mask, hidden_states, output_attentions=False):
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.config.decoder_start_token_id, device=input_embeds.device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return decoder_outputs.last_hidden_state


class ChronosBoltPipelineWithRetrieval(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048
    _aliases = ["PatchedT5Pipeline"]

    def __init__(self, model: ChronosBoltModelForForecastingWithRetrieval):
        super().__init__(inner_model=model)
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ):
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor.to(
                        device=self.model.device,
                        dtype=torch.float32,
                    ),
                    retrieved_seq=retrieved_seq,
                    distances=distances,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length]

    def predict_quantiles(
        self,
        context: torch.Tensor,
        prediction_length: int,
        quantile_levels: List[float],
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = (
            self.predict(
                context,
                prediction_length=prediction_length,
                retrieved_seq=retrieved_seq,
                distances=distances,
            )
            .detach()
            .cpu()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(training_quantile_levels):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions, q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype), dim=-1
            ).permute(1, 2, 0)
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.pop("optimization_strategy", None)
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        context_length = kwargs.pop("context_length", None)
        if context_length is not None:
            config.chronos_config["context_length"] = context_length

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecastingWithRetrieval"
            )
            class_ = ChronosBoltModelForForecastingWithRetrieval

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)

