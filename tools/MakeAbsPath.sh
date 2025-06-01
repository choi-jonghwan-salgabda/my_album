#!bin/bash

if [ $# -ne 2 ] ; then
	echo "매개변수가 적어요"
	echo "사용법은 $0 '옮길 파일리름' '옮길곳'"
	exit 1
fi


#현재 작업하는 디렉토리
gCurnPath="$( pwd -P )"
echo "gCurnPath = ${gCurnPath}"

#실행할 쉘이 있는 디렉토이 읽기
gShellPath="$( cd "$(dirname "$0")" ; pwd -P )"
echo "gShellPath = ${gShellPath}"

#기준이 되는 파일이 있는 디렉토리
gTagtPath="$1"
if [ ! -e "${gTagtPath}" ] ; then 
	echo "gTagtPath( ${gTagtPath} ) is Not Exist"
	return 99
fi

#중복하일을 찾을 디랙토리
gDestPath="$2"
if [ ! -e "${gDestPath}" ] ; then 
	echo "gDestPath( ${gDestPath} ) is Not Exist"
	return 98
fi

gTabs=""



MakeAbsPath(){
	echo "${gTab}Start MakeAbsPath() Enviroment is $( pwd -P )"

	local	pPath="$1"
	local	lEnvePath="$( pwd -P)"

#	Path의 마지막"/"를 지운다.
	local	lTemp=$(echo ${pPath}| rev | cut -c 1 | rev)
#	echo "${gTab}lTemp last Charector is (${lTemp})"
	if [ "${lTemp}" = "/" ]; then
		pPath="${pPath%/}"     #현재path의 마지막이 /이면 지운다.
	fi

#	Dir가 "." 로 시작하면 걍로조정작읍을 한다.
	lTemp=$(echo ${pPath} | cut -c 1 )
#	echo "${gTab}Param pPath is (${pPath}), Temp is (${lTemp})"
	if [ "${lTemp}" = "." ] ; then  #절대경로(HOME)
		lTemp=$(echo ${pPath} | cut -c 1-3 )
		#디렉토리가 모 디렉토리로 시작하면 모디렉토리가 없을때까지 현재 패스를 줄인다.o
		while [ "${lTemp}" = "../" ]; do
			lEnvePath="${lEnvePath%/*}"
			pPath="${pPath#../}"
#			echo "${gTab}Current Path(${lEnvePath}), Path is ${pPath}"
			lTemp=$(echo "${pPath}" | cut -c 1-3 )
		done

		if [   "$(echo "${pPath}" | cut -c 1-2 )" = "./" ]; then #디렉토리가 ./로 시작하면 /까지 지운다.
			pPath="${pPath#./}"
		elif [ "$(echo "${pPath}" | cut -c 1 )"   = "." ]; then  #디렉토리가 .으로 시작하면 .을 지운다.n
			pPath="${pPath#.}"
		fi
		echo "${lEnvePath}/${pPath}"
	elif [ "${lTemp}" = "~" ] ; then  #절대경로(HOME)
		echo "/home${pPath#~/}"
	elif [ "${lTemp}" = / ] ; then  #절대경로(HOME)
		echo "${pPath#/}"
	else
		echo "${lEnvePath}/${pPath}"
	fi
}	

AddTwoPath(){
	echo "${gTab}Start AddTwoPath() Enviroment is $( pwd -P )"

	local	pManPath="$1"
	local	pSubPath="$2"
#	echo "First Param pPath is (${pManPath}) Second Param pPath is (${pManPath})"

#	Path의 마지막"/"를 지운다.
	local	lTemp=$(echo ${pManPath}| rev | cut -c 1 | rev)
	if [ "${lTemp}" = "/" ]; then
		pManPath="${pManPath%/}"     #현재path의 마지막이 /이면 지운다.
	fi
	echo "${gTab}main Path is (${pManPath}), Temp is (${lTemp})"

#	Dir가 "." 로 시작하면 걍로조정작읍을 한다.
	lTemp=$(echo ${pSubPath} | cut -c 1 )
	echo "${gTab}Second Path is (${pSubPath}), Temp is (${lTemp})"
	if [ "${lTemp}" = "." ] ; then  #절대경로(HOME)
		lTemp=$(echo ${pSubPath} | cut -c 1-3 )
		#디렉토리가 모 디렉토리로 시작하면 모디렉토리가 없을때까지 현재 패스를 줄인다.o
		while [ "${lTemp}" = "../" ]; do
			pManPath="${pManPath%/*}"
			pSubPath="${pSubPath#../}"
			echo "Main Path is ${pManPath}, Sub Path is ${pSubPath}"
			lTemp=$(echo "${pSubPath}" | cut -c 1-3 )
		done

		if [   "$(echo "${pSubPath}" | cut -c 1-2 )" = "./" ]; then #디렉토리가 ./로 시작하면 /까지 지운다.
			pSubPath="${pSubPath#./}"
		elif [ "$(echo "${pSubPath}" | cut -c 1 )"   = "." ]; then  #디렉토리가 .으로 시작하면 .을 지운다.n
			pSubPath="${pSubPath#.}"
		fi
#	elif [ "${lTemp}" = "~" ] ; then  #절대경로(HOME)
#		pSubPath="/home/owner${pSubPath#~/}"
	elif [ "${lTemp}" = "/" ] ; then  #절대경로(HOME)
		pSubPath="${pSubPath#/}"
	fi
	echo "${pManPath}/${pSubPath}"
}	

echo "${gTab}"
#MakeAbsPath "${gTagtPath}" "${gDestPath}"
AddTwoPath "${gTagtPath}" "${gDestPath}"
