#!/usr/bin/env bash

#scripte의 오류점검 보완 https://markruler.github.io/posts/shell/minimal-safe-bash-script-template/
#set -Eeuo pipefail
set -Eeu
set -x

MkAbsPath(){
	local	pArgPath="$1"
	local	lEnvPath="$( pwd -P)"
	local	lPath="${pArgPath}"
	local	lEnvr="${lEnvPath}"
	local	lTemp=""

	#Path가"." 로 시작하면 조정작업을 한다.
	lTemp="$(echo "${pArgPath}" | cut -c 1 )"
	if [ "${lTemp}" = "~" -o "${lTemp}" = "/" ]; then #디렉토리가 ~ 나 /시작하면 그래로 사용
	  	lPath="${pArgPath#/}"
		lEnvr=""

	elif [ "$(echo "${lPath}" | cut -c 1-3 )" = "../" ]; then #디렉토리가 .으로 시작하면 .을 지운다.
		lPath="${pArgPath}"
		lEnvr="${lEnvPath}"

		#디렉토리가 모 디렉토리로 시작하면 모디렉토리가 없을때까지 현재 패스를 줄인다.o
		while [	"$(echo "${lPath}" | cut -c 1-3 )" = "../" \
			 -a "$(echo "${lEnvr}" | cut -c 1   )" = "/" ] ; do
				
				lPath="${lPath#../}"
				lEnvr="${lEnvr%/*}"
		done
		while [	"$(echo "${lPath}" | cut -c 1-3 )" = "../" ]; do
			lPath="${lPath#../}"
		done	

	elif [ $(echo "${pArgPath}" | cut -c 1-2 ) = "./" ]; then #디렉토리가 ./로 시작하면 /까지 지운다.
		lPath="${pArgPath#./}"
		lEnvr="${lEnvPath}"

	fi

	#현 환경패스의 마지막 /를 지운다.
	if [ "$(echo "${lEnvr}" | rev | cut -c 1 | rev)" = "/" ]; then
		lEnvr="${lEnvr%/}" #현재path의 마지막이 /이면 지운다.
	fi

	echo "${lEnvr}/${lPath}"
}

MkAbsPath "$1"
