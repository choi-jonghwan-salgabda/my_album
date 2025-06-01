
#!/bin/bash

if [ $# -ne 3 ] ; then
	echo "매개변수가 적어요"
	echo "사용법은 ""$0"" '기준디렉토리'"" '읽을 디렉토리'"" '중복된 파일 저장할 디렉토리'"" 입니다."
	exit 1
fi


#현재 작업하는 디렉토리
GLB_CURNT_PATH="$( pwd -P )"
echo "GLB_CURNT_PATH = ${GLB_CURNT_PATH}"

#실행할 쉘이 있는 디렉토이 읽기
GLB_SHELL_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo "GLB_SHELL_PATH = ${GLB_SHELL_PATH}"

#기준이 되는 파일이 있는 디렉토리
gBasePath="$1"
echo "gBasePath = ""${gBasePath}"

#중복하일을 찾을 디랙토리
gTargPath="$2"
echo "gTargPath = ""${gTargPath}"

#찾은 중복 파일을 저장할 디렉토리
gDestPath="$3"
echo "gDestPath = ""${gDestPath}"


function AddDirTail(){
	lPath=$1
	lDir=$2
	echo "param1 : ${lPath}, Param2 : ${lDir}"
  	if [ "${lPath:${#lPath} -1:1}" != "/" ]; then
		lPath="${lPath}"\/""
	fi
	lPath="${lPath}${lDir}"
	echo "${lPath}"
}


##디렏토리 이름만 추출-../ ./ .등은 버림
#function GetDirName(){
#	lDirName="$1"
##header	
#    	if [ "${lDirName:0:1}" != "/" ] ; then
#    		if [ "${lDirName:0:3}" = "../" ] ; then
#	  		lDirName="${lDirName:3:${#lDirName} -3}"
# 		elif [ "${DirName:0:2}" = "./" ] ; then
#  			lDirName="${lDirName:2:${#lDirName} -2}"
#  		elif [ "${lDirName:0:1}" = "." ] ; then
#  			lDirName="${lDirName:1:${#lDirName} -1}"
#  		fi
#	fi
#
##tailer
#  	if [ "${lDirName:${#lDirName} -1:1}" = "/" ]; then
#		lDirName="${lDirName:0:${#lDirName} -1}"
#	fi
#	GlbRetVal="${lDirName}"
#	echo "GetDieName() : " "{lDirName}""GlbRetVal : ""${#GblRetVal}"
#}

FindReadedFiles(){
	echo "Start FineReadedFiles() Enviroment is $(pwd)"
	echo "                     Enviroment -P is $(pwd -P)"
	lFile="$1"
	echo "ReadFile : ""{lFile}"
	echo "Target Directory : ""{gTargPath}"
	echo "Destination Directory : ""{gDestPath}"
	find "${gTargPath}" \! \( \( -type d -path "lost+found" -o -type d -path "./.Trash-1000/*" -o -type d -path "$(pwd)" \) -prune \) -type f -iname "${lFile}" -print
# 	find "${gTargPath}" \! \( \( -type d -path 'lost+found' -o -type d -path './.Trash-1000/*' -o -type d -path "$(pwd)" \) -prune \) -type f -iname "${lFile}" -exec tar --remove-files -uf "${gDestPath}""/""${lFile}".tar {} +;
	echo "                     Enviroment -P is $(pwd -P)"
	echo "Ended FineReadedFiles() Enviroment is $(pwd)"
}


GoToLastDir(){
	echo "Start GoTpLastDir() Enviroment    is $(pwd)"
	echo "                    Enviroment -p is $(pwd)"
	echo "             Current Directory    is "$1

 	ls "$1" | while read lFile
	do
		if [ -d ${lFile} ];  then    #읽은  파일이 디렉토리이면
 	 	        echo "   Readed Directory  : ${lFile}"
  			if [ "${lFile}" = "." ] || [ "${lFile}" = ".." ] ; then continue  #자신의 디렉토리이거나 모 디럭토리임.
   			else
				cd "$1"
    				GoToLastDir "${lFile}"
				cd ".."
  			fi
 		else				#읽은 파일이 파일이 아니면?"
			echo "   Readed file is ${lFile}"
			if [ -f ${lFile} ]; then 
				FindReadedFiles "${lFile}" 
			else
				if [ -a ${lFile} ]; then 
					echo "   Readed Data : ${lFile} exists (deprecated)"
				elif [ -s ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is not zero size"
				elif [ -d ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a directory"
				elif [ -b ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a block device"
				elif [ -c ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a character device"
				elif [ -p ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a pipe"
				elif [ -h ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a symbolic link -h"
				elif [ -L ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a symbolic link -L"
				elif [ -S ${lFile} ]; then 
					echo "   Readed Data : ${lFile} is a socket"
				elif [ -t ${lFile} ]; then 
					echo "   Readed Data : ${lFile} (descriptor) is associated with a terminal device"
				elif [ -r ${lFile} ]; then 
					echo "   Readed Data : ${lFile} has read permission (for the user running the test)"
				elif [ -w ${lFile} ]; then 
					echo "   Readed Data : ${lFile} has write permission (for the user running the test)"
				elif [ -x ${lFile} ]; then 
					echo "   Readed Data : ${lFile} has execute permission (for the user running the test)"
				elif [ -g ${lFile} ]; then 
					echo "   Readed Data : ${lFile} set-group-id (sgid) flag set on file or directory"
				elif [ -u ${lFile} ]; then 
					echo "   Readed Data : ${lFile} set-user-id (suid) flag set on file"
				elif [ -k ${lFile} ]; then 
					echo "   Readed Data : ${lFile} sticky bit set"
				elif [ -O ${lFile} ]; then 
					echo "   Readed Data : ${lFile} you are owner of file"
				fi
			fi
 		fi
	done


	echo "             Current Directory    is "$1
	echo "                    Enviroment -p is $(pwd)"
	echo "Ended GoTpLastDir() Enviroment    is $(pwd)"
}

GoToLastDir "${gBasePath}"

