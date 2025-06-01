

#!/usr/bin/env bash

#scripte의 오류점검 보완 https://markruler.github.io/posts/shell/minimal-safe-bash-script-template/
#set -Eeuo pipefail
set -Eeu

gShellPAth=""
gBasePAth=""
gTagtPAth=""
gDestPAth=""
gDbgLvl=0 #debug 할 수준 1,2,3
gIsRun=1
gTabs=""

usage() {
  cat <<EOF
	Usage: $(basename "$0") 조건(Option) [parameter]

 	설명:
 	이 스크립트는 기본 기분디렉토리의 파일을 찾을 디렉토리에서 찾아 같은것(이름, 크기, 만든시각)이
 	있으면 지우고 이름만 같은것은 롦길곳으로 옮겨줍니다.

	Available options:

	-h, --help      Print this help and exit
	-v, --verbose   Print script debug info
	-f, --flag      Some flag description
	-l, --level     오류만 추적하기위한 메세지 출력 수준 0,1,2,3
	-t, --test      오류만 추적하기 : 실제 파일의 이동은 없음
	-B, --base      기준이되는 파일의 위치
	    예) "$0" -B ./base
	-T, --target    찾을곳, 찾을 대상 디렉토리
	    예) "$0" -T ./target
	-D, --destance  찾을곳에서 찾은 파일중 같은 파일을 롦길곳, 목적지 디렉토리
	    예) "$0" -D ./destance
EOF
  exit
}

if [ $# -lt 3 ] ; then
	usage
	exit
fi

EchoMsg(){
	local pDbugLevel=$1
	local pStartEnd="$2"
	local pMessage="$3"
	local lTabStr="  "

	if [ "${pStartEnd}" = " " ] ; then
		if [ $gDbgLvl -ge $pDbugLevel ]; then
			echo "${gTabs}${pMessage}"
		fi
	elif [ "${pStartEnd}" = "S" ] ; then
		gTabs="${gTabs}${lTabStr}"
		if [ $gDbgLvl -ge $pDbugLevel ]; then
			echo "${gTabs}${pMessage}"
		fi
		gTabs="${gTabs}${lTabStr}"
	elif [ "${pStartEnd}" = "E" ] ; then
		gTabs="${gTabs%${lTabStr}}"
		if [ $gDbgLvl -ge $pDbugLevel ]; then
			echo "${gTabs}${pMessage}"
		fi
		gTabs="${gTabs%${lTabStr}}"
	fi

}

parse_params() {
  # default values of variables set from params
  flag=0

  while :; do
    case "${1-}" in
    -h | --help) usage 
  		EchoMsg 0 " " "${1-} Help usages"
		;;
    -v | --verbose) set -x 
  		EchoMsg 0 " " "${1-} Set -x"
		;;
    --no-color) NO_COLOR=1 ;;
    -f | --flag) flag=1  # example flag
  		EchoMsg 0 " " "${1-} Flag=${flag}"
		;;
	-l | --level) gDbgLvl=${2-}
  		EchoMsg 0 " " "Level ${gDbgLvl} Befor Shift"
		shift
  		EchoMsg 0 " " "Level ${gDbgLvl} After Shift"
		;;
	-t | --test) gIsRun=0
  		EchoMsg 0 " " "Debuging mod"
		;;
    -B | --base) # example named parameter
		#기준이 되는 파일이 있는 디렉토리
		gBasePath="${2-}"
		if [ ! -e "${gBasePath}" ] ; then 
  			EchoMsg 0 " " "gBasePath( ${gBasePath} ) is Not Exist"
			return 99
		else
  			EchoMsg 0 " " "gBasePath( ${gBasePath} )"
		fi
		shift
		;;
    -T | --target) # example named parameter
		#중복파일을 찾을 디랙토리
		gTagtPath="${2-}"
		if [ ! -e "${gTagtPath}" ] ; then 
  			EchoMsg 0 " " "gTagtPath( ${gTagtPath} ) is Not Exist"
			return 99
		else
  			EchoMsg 0 " " "gTagtPath( ${gTagtPath} )"
		fi
		shift
		;;
    -D | --param) # example named parameter
		#찾은 중복 파일을 둘곳
		gDestPath="${2-}"
		if [ ! -e "${gDestPath}" ] ; then 
			mkdir -p "${gDestPath}" 
			if [ $? -ne 0 ];then
				EchoMsg 0 " " "MoveFIle(): Failed Mkdir Comd ${gDestPath}"
				return 99
			fi
  			EchoMsg 0 " " "gDestPath( ${gDestPath} )"
		elif [ -f "${gDestPath}" ]; then
			EchoMsg 0 " " "Move File(): ${gDestPath} is Exist File"
			exit
		fi
		shift
		;;
    -?*) EchoMsg 0 " "  "Unknown option: ${1-}" ;;
    *) break ;;
    esac
    shift
  done

  return 0
}

MakeAbsPath(){
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

AddTwoPath(){

	local	pManPath="$1" 
	local	pSubPath="$2"
	local	lTemp1="$(echo "${pManPath}" | rev | cut -c 1 | rev)"
	local	lTemp2="$(echo "${pSubPath}" | cut -c 1 )"

	#main Path의 마지막"/"를 지운다.
	if [ "${lTemp1}" = "/" -a "${lTemp2}" = "/" ]; then
		echo "${pManPath%/}${pSubPath}"

	elif [ "${lTemp1}" != "/" -a "${lTemp2}" != "/" ]; then
		echo "${pManPath}/${pSubPath}"
	else
		echo "${pManPath}${pSubPath}"
	fi
}	

MkWildCard(){
	local pFile="$1"
	local lFIle=""

	if  [ $(echo ${pFile} | grep '\.') ] ; then
		lFile="$(echo ${pFile} | sed 's/\./\*\./1')"
	else
		lFile="${pFile}*"
	fi
	echo "${lFile}"
}

#받은 파일을 목적지에서 찾아 옮긴다.
#파일이 존재하면 같은아름의 유사 파일 갯수만큼 번호를 붙인다.
#목적지 디렉토리가 없으면 만든다.
#목적지가 파일이면 종료한다.
MoveFile(){

	local	pTagtFull="$1"
	local 	lTagtPath="$(dirname  "${pTagtFull}")"	# Extract Abs Path 
	local 	lTagtFile="$(basename "${pTagtFull}")"	# Extract filename 
	local	lTempPath=""
	local	Cnt=0

	EchoMsg 1 "S" "Start MoveFile ${pTagtFull} Move To ${gDestPath}"

	#파일을 이름과 확장자를 나누기
	local lFname=""
	local lFexte=""
	if  [ $(echo ${lTagtFile} | grep '\.') ] ; then
		lFname="$(echo ${lTagtFile} | sed 's/\..*//1')"
		lFexte="$(echo ${lTagtFile} | sed 's/.*\.//1')"
	else
		lFname="${lTagtFile}*"
	fi

	EchoMsg 2 " " "MoveFile():      File ${lFname}.${lFexte}"

	lTempPath="$(AddTwoPath "${gDestPath}" "${lFname}*.${lFexte}")"
	lCnt="$(ls -al ${lTempPath} | grep '^-' | wc -l)" #같은이름의 파일수 구하기
	if [ ${lCnt} -gt 0 ] ;  then
		lTempPath="${lTagtPath}"/"${lFname}(${lCnt}).${lFexte}"
	else
		lTempPath="${lTagtPath}"/"${lFname}.${lFexte}"
	fi

	#파일을 옮긴다
	if [ ${gIsRun} = 1 ]; then
		mv "${lTempPath}" "${gDestPath}"
		if [ $? -ne 0 ]; then
	 		EchoMsg 0 " " "MoveFile(): Failed File ${lTempPath} Move To ${gDestPath}"
			return 99
		fi
	else
		EchoMsg 0 " "     "MoveFile():        Move ${lTempPath} To ${gDestPath}"
    fi

	EchoMsg 1 "E" "Ended MoveFile(): Will Move ${lTempPath} To ${gDestPath}"
}

MvOrDelFile(){
	local	pTagtFull="$1"
	local 	lTagtPath="$(dirname  "${pTagtFull}")"			# Extract Abs Path 
	local 	lTagtFile="$(basename "${pTagtFull}")"			# Extract filename 
	local	lTempPath=""

	EchoMsg 1 "S" "Start MvOrDelFile(): File ${pTagtFull} Move To ${gDestPath}"

	lTempPath="$(AddTwoPath "${gDestPath}" "${lTagtFile}")"
	EchoMsg 2 " " "MvOrDelFile(): ABS  (${lTempPath})"

	if [ -e "${lTempPath}" ]; then
		if [  $( wc -c "${pTagtFull}"		| awk '{print $1 }') \
			= $( wc -c "${lTempPath}"	| awk '{print $1 }') ] ; then
			#크기가 같은것이 있으면 Target을 지운다
			if [ ${gIsRun} = 1 ]; then
				rm "${pTagtFull}"
				if [ $? -ne 0 ];then
					EchoMsg 0 "E" "Ended MvOrDelFile(): NonRm ${pTagtFull}"
				fi
				EchoMsg 3 " " "MvOrDelFile(): Rm Target  ${pTagtFull}"
			else
				EchoMsg 0 " " "MvOrDelFile():     Remove ${pTagtFull}"
			fi
		fi
	else
		MoveFile "${pTagtFull}"
	fi
	EchoMsg 1 "E" "Ended MvOrDelFile()"
}

FindReadedFiles(){
	local	pFile="$1"
	EchoMsg 1 "S" "Start FindReadedFile(): Will Find (${pFile}) in ${gTagtPath}"

	local	lFile=""
	#라인단위로 처리하므로 파일이름에 공백이 있어도 잘됨.
	find "${gTagtPath}" \! \( \(   -type d -path "$(pwd)" \
								-o -type d -path "*/최종환개인자료*" \
								-o -type d -path "*/System Volume Information*" \
								-o -type d -path "*/lost+found*" \
								-o -type d -path "*/.Trash-1000" \) -prune \) \
								-type f -iname "${pFile}" | while read lFile ;
	do
		MvOrDelFile "${lFile}"
	done

	EchoMsg 1 "E" "Ended FindReadedFile():    Finded (${pFile}) in ${gTagtPath}"
}

GoToLastDir(){
	local	pCurDir="$1"
	local	lNewCurntPath="$( MakeAbsPath "${pCurDir}" )"
	EchoMsg 1 "S" "Start GotoLastDir(): Enviroment is  $(pwd -P) / PCurDir=${pCurDir}"


	if [ ! -e "${lNewCurntPath}" ] ; then 
		EchoMsg 0 "E" "Ended GotoLastDir(): Not Exist Dir  ${lNewCurntPath}"
		return 100
	fi
	

	cd "${lNewCurntPath}"
	if [ $? -ne 0 ];then
		EchoMsg 0 " " "GotoLastDir(): Failed Cd Comd ${lNewCurntPath}"
   	fi
	EchoMsg 2 " " "GotoLastDir(): New Path is    $( pwd -P )"

	ls "./" | while read lFile ; #라인단위로 처리하므로 파일이름에 공백이 있어도 잘됨.
	do
		EchoMsg 3 " " "GotoLastDir(): Readed file    ${lFile}"

		if [ -d "${lFile}" ] ; then #읽은 파일이 디렉토리이면
  			if [ "${lFile}" != "." -a "${lFile}" != ".." ] ; then
    			GoToLastDir "${lFile}"
  			fi
 		elif [ -f "${lFile}" ] ; then #읽은 파일이 파일이면" 
			FindReadedFiles "${lFile}"
		elif [ -e "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is a Exist"
		elif [ -s "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is not zero size"
		elif [ -z "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Null"
		elif [ -n "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Not Null"
		elif [ -c "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Not Null"
		elif [ -b "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Block Device"
		elif [ -p "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Pipe"
		elif [ -h "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Hard Link"
		elif [ -L "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is Symbolic"
		elif [ -S "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is File Socket"
		elif [ -t "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is 파일 디스크립터가 터미널 디바이스와 연관됨"
		elif [ -r "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} has read permission (for the user running the test)"
		elif [ -w "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} has write permission (for the user running the test)"
		elif [ -x "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} has execute permission (for the user running the test)"
		elif [ -O "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is 존재하며 현재 소유주가 맞는지 확인"
		elif [ -G "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} 파일이 존재하고 그룹사용자가 맞는지 확인"
		elif [ -k "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is 고정비트가 설정되어 있는지 확인"
		elif [ -u "${lFile}" ] ; then 
			EchoMsg 0 " " "GotoLastDir(): Readed File ${lFile} is SUIO가 설정되어 있는지 확인"
		else
			EchoMsg 0 " " "GotoLastDir(): Readed File Readed file is what is ${lFile}"
 		fi
	done
	EchoMsg 2 " " "GotoLastDir(): End Read in New$( pwd -P )"

	cd ".."
	if [ $? -ne 0 ];then
		EchoMsg 0 " " "GotoLastDir(): Cd Command Failed for File(..)"
   	fi

	EchoMsg 1 "E" "Ended GotoLastDir(): Enviroment is  $(pwd -P)"
}

parse_params "$@"

#현재 작업하는 디렉토리
gCurnPath="$( pwd -P )"
EchoMsg 0 " " "gCurnPath = ${gCurnPath}"

#실행할 쉘이 있는 디렉토이 읽기
gShellPath="$( cd "$(dirname "$0")" ; pwd -P )"
EchoMsg 0 " " "gShellPath = ${gShellPath}"


#기준파일이 있는 곳
EchoMsg 0 " " "gBasePath = ${gBasePath}"

#찾을파일이 있는곳
EchoMsg 0 " " "gTagtPath = ${gTagtPath}"

#중복된파일 옮길곳
EchoMsg 0 " " "gDestPath = ${gDestPath}"

#로그에 시작하기 위해 표시
EchoMsg 0 " " "${gTabs}"

GoToLastDir "${gBasePath}"

