#!/bin/bash
FILE=/var/lib/jenkins/jobs/devops/workspace/test1.html
grep -i Version $FILE | awk '{print $5 " " $6}'
