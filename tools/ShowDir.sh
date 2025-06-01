BASE_PATH=$1
TARGET_PATH=$2
echo $BASE_PATH
echo $TARGET_PATH

find ${BASE_PATH} -maxdepth 1 ! \( -type d \( -path 'lost+found' -o -path './.Trash-1000/*'  \) -prune \) -type d | while read dirname1

do
	cd "${dirname1}"
	SHELL_PATH=`pwd -P`
	echo $SHELL_PATH
	sh ~/CurDirBaseDelFileInParam1.sh ${TARGET_PATH}
	cd "${BASE_PATH}"
	SHELL_PATH=`pwd -P`
	echo $SHELL_PATH
done
