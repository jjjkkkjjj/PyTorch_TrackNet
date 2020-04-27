#!/bin/bash
# usage: bash access_colab.sh https://~~
echo "Note that you should type 'chmod 755 access_colab.sh' first"

URL=$1
for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open $URL
  sleep 3600
done

