#!/bin/bash
export PROJECT=`basename $PWD`
docker build $@ \
 --build-arg  USER=`id -un` --build-arg UID=`id -u` \
 --build-arg GROUP=`id -gn` --build-arg GID=`id -g` \
        --tag $USER/$PROJECT - < ./Dockerfile
