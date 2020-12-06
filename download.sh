#!/usr/bin/env bash

set -e

wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip -O ./Tennis_Linux.zip
wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip -O ./Tennis_Linux_NoVis.zip
unzip ./Tennis_Linux.zip -d ./
unzip ./Tennis_Linux_NoVis.zip -d ./
