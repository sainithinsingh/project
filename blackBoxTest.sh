#!/bin/bash

HOSTS=`cat /tmp/testIP`

for i in $HOSTS
do
 ping -q -c2 $i > /dev/null
 if [ $? -eq 0 ]
 then
   echo $i " Ping : Success"
 else
   echo $i " Ping : Fail"
 fi
done

ping -q -c2 youtube.com > /dev/null
if [ $? -eq 0 ]
then
   echo "Ping youtube : Success "
else
   echo "Ping youtube : Fail "
fi

unset http_proxy https_proxy

for host in $HOSTS 
do 
	`wget --quiet http://$host -O index.html`
	file=index.html
	tidy -f Output -quiet -error -ashtml $file	
	if [ $? != 2 ]
	then
	  echo "html format Check : Success"
	else
	  echo "html format Check : Fail"
	fi
	grep -i Version index.html | awk '{print $5 " " $6}'
done 

rm Output
export http_proxy=www-proxy.ericsson.se
export https_proxy=www-proxy.ericsson.se
