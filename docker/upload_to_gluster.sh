#!/bin/bash
rsync ./hythe_latest.sif 8gpu:/mnt/glusterdata/home/$1/images/hythe_latest.sif -a -v -z -P
