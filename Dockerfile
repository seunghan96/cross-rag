FROM nvcr.io/nvidia/pytorch:24.04-py3

ARG USER
ARG GROUP
ARG UID
ARG GID

# Install Dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get -y install nodejs
RUN apt update
RUN apt install -y \
    libnss3 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    fonts-liberation \
    xdg-utils \
    wget
RUN apt install -y sudo

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y ./google-chrome-stable_current_amd64.deb \
    && rm google-chrome-stable_current_amd64.deb

# Add User and Group
RUN groupadd --gid $GID $GROUP && \
    useradd --uid $UID $USER -g $GROUP --create-home && \
    usermod -aG sudo $USER && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USER:$GROUP
WORKDIR /home/$USER
ENV TZ "UTC-9"
ENV PATH "/home/$USER/.local/bin:$PATH"
ENV TORCH_HOME "/home/$USER/data/torch"
ENV PYTHONPATH "/home/$USER/workspace"

# Install Python Packages
RUN pip install --upgrade pip
RUN pip install --upgrade gpustat
RUN pip install --upgrade pandas
RUN pip install --upgrade tqdm
RUN pip install --upgrade scipy
RUN pip install --upgrade scikit-image
RUN pip install --upgrade matplotlib
RUN pip install --upgrade seaborn
RUN pip install --upgrade "jsonargparse[all]"
RUN pip install --upgrade fredapi
RUN pip install --upgrade prismstudio
RUN pip install --upgrade holidays
RUN pip install --upgrade uvicorn
RUN pip install --upgrade fastapi
RUN pip install --upgrade requests
RUN pip install --upgrade jinja2
RUN pip install --upgrade httpx
RUN pip install --upgrade slack_sdk
RUN pip install --upgrade selenium
RUN pip install --upgrade openai
RUN pip install --upgrade pymongo

# Set Working Directory
WORKDIR /home/$USER/workspace

 