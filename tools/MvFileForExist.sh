
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
#	echo "${gTab}Start MakeAbsPath() Enviroment is $( pwd -P )"

	local	pPath="$1""/"
	local	pDir="$2"

	local	lTemp=$(echo ${pPath}| rev | cut -c 1 | rev)
#	echo "lTemp 1 is (${lTemp})"
	if [ "${lTemp}" = "/" ]; then
		pPath="${pPath%?}"     #현재path의 마지막이 /이면 지운다.
#		echo "Local Current Path is ${pPath}"
	fi

	lTemp=$(echo ${pDir} | cut -c 1 )
#	echo "lTemp 2 is (${lTemp})"
	if [ "${lTemp}" = "~" -o "${lTemp}" = "/" ] ; then  #절대경로(HOME)
		echo "pDir(${pDir})"
		return
	else
		
		lTemp=$(echo "${pDir}" | cut -c 1-3 )
		if [ "${lTemp}" = "../" ]; then  #디렉토리가 부 디렉토리 믿의 것이면 부디렉토리가 없을때까지 현재 패스를 줄인다.
			while [ "${lTemp}" = "../" ]; do
				pPath="${pPath%"/"}"
				pDir="${pDir#"../"}"
#				echo "Local Current Path is ${pPath}, pPath is ${pDir}"
				lTemp=$(echo "${pDir}" | cut -c 1-3 )
			done
		
		elif [ "$(echo "${pDir}" | cut -c 1-2 )" = "./" ]; then #디렉토리가 ./로 시작하면 /까지 지운다.
			pDir="${pDir#"./"}"
		elif [ "$(echo "${pDir}" | cut -c 1 )"   = "." ]; then  #디렉토리가 .으로 시작하면 .을 지운다.n
			pDir="${pDir#"."}"
		fi
		echo "${pPath}/${pDir}"
	fi
}
MoveFile(){
#	echo "${gTab}Start MoveFile() File($1) Move To $2"
	gTab="${gTab}\t"
	local 	pTagtPath="$(dirname "$1")"			# Extract Abs Path 
	local 	pTagtFile="$(basename "$1")"			# Extract filename 
	local	pDestPath="$2"
	local	lFexte=""
	local	lFname=""

	echo "${gTab}Start MoveFile() File(${pTagtPath}/${pTagtFile}) Move To ${pDestPath}"

	pDestPath="$(MakeAbsPath "${pDestPath}" "${pTagtFile}")"
	echo "${gTab}Destation File(${pDestPath})"

	if [ -d "${pDestPath}" ]; then
		echo "${gTab}File(${pDestPath}) is Exist directory File"
		return 97
	elif [ -f "${pDestPath}" ]; then
		#파일을 이름과 확자자로 나누기
		echo "${gTab}File(${pDestPath}) is Exist File"
		if [ $(expr index ${pDestPath} ".") -ne 0 ]; then
			lFexte=".${pDestPath#*.}" 		# Extract extension
			lFname="${pDestPath%.*}"
			echo "${gTab}1${lFname}*${lFexte}"
		else
			lFexte=""
			lFname="${pDestPath}"
			echo "${gTab}2${lFname}*${lFexte}"
		fi
		echo "${gTab}3${lFname}*${lFexte}"
	elif [ -e "${pDestPath}" ]; then
		echo "${gTab}File(${pDestPath}) is Exist Other File"
		return 96
	fi

	lCnt="$(ls -al ${lFname}*${lFexte} | grep '^-' | wc -l)" #같은이름의 파일수 구하기
	pDestPath="${lFname}(${lCnt})${lFexte}"
	echo "${gTab}Befor Move(${pTagtPath}/${pTagtFile}) To ${pDestPath}"
	mv "${pTagtPath}/${pTagtFile}" "${pDestPath}" 

	echo "${gTab}Ended MoveFile() File(${pTagtPath}/${pTagtFile}) To ${pDestPath}"
	gTab="${gTab%??}"
}

echo "${gTab}File(${gTagtPath}) Move To ${gDestPath}"
MoveFile "${gTagtPath}" "${gDestPath}"

