#! /bin/bash

squeue -O jobid:6,jobarrayid:12,name:40,account:8,username:10,priority,qos:10,timeused:11,statecompact:4,reasonlist,tres:50,submittime

./resource_usage
