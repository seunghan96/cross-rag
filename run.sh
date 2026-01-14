#!/bin/bash
PROJECT=`basename $PWD`
WORKDIR="/home/$USER/workspace"
LABDIDIR="/lab-di"
NFSFINDATADIR="/nfs-fin"
GPU=${GPU:-all}


if [[ $1 == "jupyter" ]]; then
        NAME="${NAME:-$1}"
        OPTIONS="${OPTIONS} -p $PORT:8888"
        AFTER="--NotebookApp.password=1234 --NotebookApp.token=1234"
elif [[ $1 == "tensorboard" ]]; then
        NAME="${NAME:-$1}"
        OPTIONS="${OPTIONS} -p $PORT:6006"
elif [[ $1 == "bash" ]]; then
        NAME="${NAME:-bash}"
        OPTIONS="${OPTIONS} -p 8234:8000"
else
    NAME="${NAME:-$(TZ=UTC-9 date +%y%m%d%H%M%S)-$(openssl rand -hex 4)}"
fi

# docker run -it --rm --user root \
docker run -it --rm --user seunghan.lee \
        -v $LABDIDIR:/home/$USER/lab-di \
        -v $NFSFINDATADIR:/home/$USER/nfs-fin \
        -v `pwd`:$WORKDIR \
        -v /nfsdata/home/$USER/.cache:/home/$USER/.cache \
        --shm-size=512G ${OPTIONS} \
        --gpus $GPU \
        --name ${USER}-${PROJECT}-${NAME} ${USER}/${PROJECT} \
        $@ $AFTER