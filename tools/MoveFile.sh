
#!/usr/bin/env bash

#scripte의 오류점검 보완 https://markruler.github.io/posts/shell/minimal-safe-bash-script-template/
#set -Eeuo pipefail
set -Eeu
set -x

gDestPath="~/sambaData/ShareData/TempWllDel/"
MoveFile(){

	local	pTagtFull="$1"
	local 	lTagtPath="$(dirname  "${pTagtFull}")"			# Extract Abs Path 
	local 	lTagtFile="$(basename "${pTagtFull}")"			# Extract filename 
	local	lTempPath=""
	local	Cnt=0

##4$	EchoMsg 1 "S" "Start MoveFile ${pTagtFull} Move To ${gDestPath}"

	#파일을 이름과 확장자를 나누기
	local lFname=""
	local lFexte=""
	if  [ $(echo ${lTagtFile} | grep '\.') ] ; then
		lFname="$(echo ${lTagtFile} | sed 's/\..*//1')"
		lFexte="$(echo ${lTagtFile} | sed 's/.*\.//1')"
	else
		lFname="${lTagtFile}*"
	fi

#	EchoMsg 2 " " "MoveFile():      File ${lFname}.${lFexte}"

	lTempPath="$(AddTwoPath "${gDestPath}" "${lFname}*.${lFexte}")"
	lCnt="$(ls -al ${lTempPath} | grep '^-' | wc -l)" #같은이름의 파일수 구하기
	if [ ${lCnt} -gt 0 ] ;  then
		lTempPath="${lTagtPath}"/"${lFname}(${lCnt}).${lFexte}"
	else
		lTempPath="${lTagtPath}"/"${lFname}.${lFexte}"
	fi

	#파일을 옮긴다
#	if [ ${gIsRun} ]; then
#		mv "${lTempPath}}" "${gDestPath}"
#		if [ $? -ne 0 ]; then
#	 		EchoMsg 0 " " "MoveFile(): Failed File ${lTempPath} Move To ${gDestPath}"
#			return 99
#		fi
#	else
#		EchoMsg 0 " "     "MoveFile():        Move ${lTempPath} To ${gDestPath}"
#    fi

#	EchoMsg 1 "E" "Ended MoveFile(): Will Move ${lTempPath} To ${gDestPath}"
}
MoveFile "$1"
