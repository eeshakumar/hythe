#!/bin/bash
# $1 configuration, $2 user, $3 goal name
echo "Building $.."
bazel build //configuration/src:$1

echo "Uploading.."
rsync bazel-bin/configuration/src/$1* 8gpu:/mnt/glusterdata/home/$2/$3 -a --copy-links -v -z -P
#rsync bazel-bin/configuration/${1}.runfiles 8gpu:/mnt/glusterdata/home/$2/$3/$1.runfiles -a --copy-links -v -z -P
rsync ./gluster/run.sh 8gpu:/mnt/glusterdata/home/$2/$3/run.sh -a --copy-links -v -z -P