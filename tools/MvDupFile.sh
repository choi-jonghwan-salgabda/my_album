

#!bin/bash

if [ $# -ne 3 ] ; then
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
gBasePath="$1"
if [ ! -e "${gBasePath}" ] ; then 
	echo "gBasePath( ${gBasePath} ) is Not Exist"
	return 99
fi

#중복하일을 찾을 디랙토리
gTagtPath="$2"
if [ ! -e "${gTagtPath}" ] ; then 
	echo "gTagtPath( ${gTagtPath} ) is Not Exist"
	return 99
fi

#찾은 중복 파일을 둘곳
gDestPath="$3"
if [ ! -e "${gDestPath}" ] ; then 
	echo "gDestPath( ${gDestPath} ) is Not Exist"
	return 98
fi

gTabs=""
gTabStr="=="



MakeAbsPath(){
#	echo "${gTabs}Start MakeAbsPath() Enviroment is $( pwd -P )"

	local	pPath="$1"
	local	lEnvePath="$( pwd -P)"

	local	lTemp=$(echo ${lEnvepath}| rev | cut -c 1 | rev)
	if [ "${lTemp}" = "/" ]; then
		lEnvepath="${lEnvepath%/}"     #현재path의 마지막이 /이면 지운다.
	fi
#	Path가"." 로 시작하면 조정작업을 한다.
	local	lTemp=$(echo ${pPath} | cut -c 1 )
#	echo "${gTabs}Param pPath is (${pPath}), Temp is (${lTemp})"
	if [ "${lTemp}" = "~" ] ; then  #절대경로(HOME)
		echo "/home/owner/${pPath#~/}"
	elif [ "${lTemp}" = / ] ; then  #절대경로(HOME)
		echo "${pPath}"
	elif [ "${lTemp}" = "." ] ; then  #절대경로(HOME)
		lTemp=$(echo ${pPath} | cut -c 1-3 )
		#디렉토리가 모 디렉토리로 시작하면 모디렉토리가 없을때까지 현재 패스를 줄인다.o
		while [ "${lTemp}" = "../" ]; do
			pPath="${pPath#../}"
			lEnvePath="${lEnvePath%/*}"
#			echo "${gTabs}Current Path(${lEnvePath}), Path is ${pPath}"
			lTemp=$(echo "${pPath}" | cut -c 1-3 )
		done

		if [   "$(echo "${pPath}" | cut -c 1-2 )" = "./" ]; then #디렉토리가 ./로 시작하면 /까지 지운다.
			pPath="${pPath#./}"
		elif [ "$(echo "${pPath}" | cut -c 1 )"   = "." ]; then  #디렉토리가 .으로 시작하면 .을 지운다.
			pPath="${pPath#.}"
		fi
		echo "${lEnvePath}/${pPath}"
	else
		echo "${lEnvePath}/${pPath}"
	fi
}	

AddTwoPath(){
#	echo "${gTabs}Start AddTwoPath() Enviroment is $( pwd -P )"

	local	pManPath="$1" 
	local	pSubPath="$2"
#	echo "First Param pPath is (${pManPath}) Second Param pPath is (${pManPath})"

#	main Path의 마지막"/"를 지운다.
	local	lTemp=$(echo ${pManPath}| rev | cut -c 1 | rev)
	if [ "${lTemp}" = "/" ]; then
		pManPath="${pManPath%/}"     #현재path의 마지막이 /이면 지운다.
	fi
#	echo "${gTabs}main Path is (${pManPath}), Temp is (${lTemp})"

#	Sub Path가 "." 로 시작하면 걍로조정작읍을 한다.
	lTemp=$(echo ${pSubPath} | cut -c 1 )
#	echo "${gTabs}Second Path is (${pSubPath}), Temp is (${lTemp})"
	if [ "${lTemp}" = "~" ] ; then  #절대경로(HOME)
		echo pSubPath="/home/owner/${pSubPath#~/}"
	elif [ "${lTemp}" = "/" ] ; then  #절대경로(HOME)
		echo pSubPath="${pSubPath}"
	elif [ "${lTemp}" = "." ] ; then  #절대경로(HOME)
		lTemp=$(echo ${pSubPath} | cut -c 1-3 )
		#디렉토리가 모 디렉토리로 시작하면 모디렉토리가 없을때까지 현재 패스를 줄인다.o
		while [ "${lTemp}" = "../" ]; do
			pSubPath="${pSubPath#../}"
			pManPath="${pManPath%/*}"
#			echo "Main Path is ${pManPath}, Sub Path is ${pSubPath}"
			lTemp=$(echo "${pSubPath}" | cut -c 1-3 )
		done

		if [   "$(echo "${pSubPath}" | cut -c 1-2 )" = "./" ]; then #디렉토리가 ./로 시작하면 /까지 지운다.
			pSubPath="${pSubPath#./}"
		elif [ "$(echo "${pSubPath}" | cut -c 1 )"   = "." ]; then  #디렉토리가 .으로 시작하면 .을 지운다.n
			pSubPath="${pSubPath#.}"
		fi
		echo "${pManPath}/${pSubPath}"
	else
		echo "${pManPath}/${pSubPath}"
	fi
}	

MoveFile(){
#	gTabs="${gTabs}${gTabStr}"
#	echo "${gTabs}Start MoveFile() File($1) Move To $2"
	local 	pTagtPath="$(dirname "$1")"			# Extract Abs Path 
	local 	pTagtFile="$(basename "$1")"			# Extract filename 
	local	pDestPath="$2"
	local	lFexte=""
	local	lFname=""

#	echo "${gTabs}Start MoveFile() File(${pTagtPath}/${pTagtFile}) Move To ${pDestPath}"

	pDestPath="$(AddTwoPath "${pDestPath}" "${pTagtFile}")"
#	echo "${gTabs}Destation File(${pDestPath})"

	if [ -d "${pDestPath}" ]; then
		echo "${gTabs}File(${pDestPath}) is Exist directory File"
		return 97
	elif [ -e "${pDestPath}" -a ! -f "${pDestPath}" ]; then
		echo "${gTabs}File(${pDestPath}) is Exist Other File"
		return 96
	elif [ -f "${pDestPath}" ]; then
		#파일을 이름과 확자자로 나누기
#		echo "${gTabs}File(${pDestPath}) is Exist File"
		if [ $(expr index "${pDestPath}" ".") -ne 0 ]; then
			lFname="${pDestPath%.*}"
			lFexte="${pDestPath#${lFname}}" 		# Extract extension
#			echo "${gTabs}1${lFname}*${lFexte}"
		else
			lFexte=""
			lFname="${pDestPath}"
#			echo "${gTabs}2${lFname}*${lFexte}"
		fi
#		echo "${gTabs}3${lFname}*${lFexte}"
		lCnt="$(ls -al ${lFname}*${lFexte} | grep '^-' | wc -l)" #같은이름의 파일수 구하기
		pDestPath="${lFname}(${lCnt})${lFexte}"
	fi
#	echo "${gTabs}Befor Move(${pTagtPath}/${pTagtFile}) To ${pDestPath}"
	mv "${pTagtPath}/${pTagtFile}" "${pDestPath}" 
	if [ $? -ne 0 ];then
		echo "${gTabs}MoveFile() Failed for File(${pTagtPath}/${pTagtFile}) Move To ${pDestPath}"
    fi
#	echo "${gTabs}Ended MoveFile() File(${pTagtPath}/${pTagtFile}) Move To ${pDestPath}"
#	gTabs="${gTabs%${gTabStr}}"
}


FindReadedFiles(){
#	gTabs="${gTabs}${gTabStr}"
#	echo "${gTabs}Start FineReadedFiles() Enviroment is $(pwd -P)"
	local	pFile="$1"
#	echo "${gTabs}ReadFile 			: ${pFile}"
#	echo "${gTabs}Target Directory 		: ${gTagtPath}"
#	echo "${gTabs}Destination Directory 	: ${gDestPath}"

	local	lFile=""
	find "${gTagtPath}" \! \( \( -type d -path "lost+found" -o -type d -path "./.Trash-1000/*" -o -type d -path "$(pwd)" \) -prune \) -type f -iname "${pFile}" | while read lFile ;  #라인단위로 처리하므로 파일이름에 공백이 있어도 잘됨.
	do
		MoveFile "${lFile}" "${gDestPath}"
	done
	if [ $? -ne 0 ];then
		echo "${gTabs}FindReadedFiles() Failed for File(${pTagtPath}/${pTagtFile}) Move To ${pDestPath}"
    fi

	pFile="${pFile}.tar"
	lFile=""
	find "${gTagtPath}" \! \( \( -type d -path "lost+found" -o -type d -path "./.Trash-1000/*" -o -type d -path "$(pwd)" \) -prune \) -type f -iname "${pFile}" | while read lFile ;  #라인단위로 처리하므로 파일이름에 공백이 있어도 잘됨.
	do
		MoveFile "${lFile}" "${gDestPath}"
	done
	if [ $? -ne 0 ];then
		echo "${gTabs}FindReadedFiles() Failed for File(${pTagtPath}/${pTagtFile}) Move To ${pDestPath}"
    fi
#	echo "${gTabs}Ended FineReadedFiles() Enviroment is $(pwd -P)"
#	gTabs="${gTabs%${gTabStr}}"
}

GoToLastDir(){
	gTabs="${gTabs}${gTabStr}"
	local	pCurDir="$1"

	local	lOldCurPath="$( pwd -P )"
	local	lNewCurntPath="$( MakeAbsPath "${pCurDir}" )"

	echo "${gTabs}Start GoTpLastDir() Enviroment is (${lOldCurPath}), Dir is ${pCurDir}"

	if [ ! -e "${lNewCurntPath}" ] ; then 
		echo "( ${lNewCurntPath} ) is Not Existi. Change Directory is Faild"
		return 100
	fi
	
#	echo "${gTabs}Will Go to New Path  is (${lNewCurntPath})"

	cd "${lNewCurntPath}"
	if [ $? -ne 0 ];then
		echo "${gTabs}Cd Command Failed for File(${lNewCurntPath}"
   	fi
#	echo "${gTabs}${pCurDir} Current New Path is $( pwd -P )"

	ls "./" | while read lFile ;  #라인단위로 처리하므로 파일이름에 공백이 있어도 잘됨.
	do
#	 	echo "${gTabs}Readed file  : ${lFile}"
		if [ -d "${lFile}" ] ;  then    #읽은  파일이 디렉토리이면
  			if [ "${lFile}" != "." -a "${lFile}" != ".." ] ; then
    				GoToLastDir "${lFile}"
  			fi
 		elif [ -f "${lFile}" ] ; then   #읽은 파일이 파일이면" 
			FindReadedFiles "${lFile}" 
		elif [ -e "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is a Exist"
		elif [ -s "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is not zero size"
		elif [ -z "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Null"
		elif [ -n "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Not Null"
		elif [ -c "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Charector Device"
		elif [ -b "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Block Device"
		elif [ -p "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Pipe"
		elif [ -h "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Hard Link"
		elif [ -L "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is Symbilic"
		elif [ -S "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is File Socket"
		elif [ -t "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} is 파일 디스크립터가 터미널 디바이스와 연관됨"
		elif [ -r "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} has read permission (for the user running the test)"
		elif [ -w "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} has write permission (for the user running the test)"
		elif [ -x "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} has execute permission (for the user running the test)"
		elif [ -O "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} 존재하며 현재 소유주가 맞는지 확인"
		elif [ -G "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} 파일이 존재하고 그룹사용자가 맞는지 확인"
		elif [ -k "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} 고정비트가 설정되어 있는지 확인"
		elif [ -u "${lFile}" ] ; then 
			echo "${gTabs}Readed File : ${lFile} SUIO가 설정되어 있는지 확인"
		else
			echo "${gTabs}   Readed file is what is ${lFile}"
 		fi
	done
#	echo "${gTabs}Goto ${pDir} End Read NewPath is - PWD $( pwd -P )"
	cd ".."
	if [ $? -ne 0 ];then
		echo "${gTabs}Cd Command Failed for File(..)"
   	fi
	echo "${gTabs}Ended GoTpLastDir() Enviroment is ($( pwd -P)), Dir is ${pCurDir}"
	gTabs="${gTabs%${gTabStr}}"
}

gTagtPath=$( MakeAbsPath "${gTagtPath}" )
gDestPath=$( MakeAbsPath "${gDestPath}" )
echo "gBasePath( ${gBasePath} )"
echo "gTagtPath( ${gTagtPath} )"
echo "gDestPath( ${gDestPath} )"
echo "${gTabs}"
GoToLastDir  "${gBasePath}"

