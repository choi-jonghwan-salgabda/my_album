
#!/bin/bash

SHELL_ARG_NUM=$#
GLB_SHELL_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
GLB_DESTI_PATH="$2"
GLB_TARGT_PATH="$1"
GLB_CURNT_PATH= pwd
Glb_Cnt_GoToLastDir=0
Glb_Cnt_DelDupFile=0
Glb_Counter_Value=1
Glb_Cnt_DirDeps=0

DelDupFile(){
	TARGT_PATH="$1"
	CURNT_PATH="$2"
#	echo "      start  Function DelDupFile"
#	echo "         Target path is  : ""${TARGT_PATH}"
	echo "         Current path is : ""${CURNT_PATH}"
	for filename in $(ls  "${CURNT_PATH}") ; do
		if [ -f "${filename}" ];  then
 			find "${TARGT_PATH}" \! \( \( -type d -path 'lost+found' -o -type d -path './.Trash-1000/*' -o -type d -path "${CURNT_PATH}" \) -prune \) -type f -iname "${filename}" -exec tar --remove-files -uf "${GLB_DESTI_PATH}""/""${filename}".tar {} +;
		fi
	done
#	echo "      Ended  Function DelDupFile"
}

GoToLastDir(){
	TARGT_PATH="$1"
	CURNT_PATH="${TARGT_PATH}"
#	echo ""
	Glb_Cnt_GoToLastDir=$((Glb_Cnt_GoToLastDir + Glb_Counter_Value))
#	echo "Start OF FOr Do Loop in GoToLastDir"
#	echo "   Call Count     : ""${Glb_Cnt_GoToLastDir}""Th"
#	echo "   Directory Deps : ""${Glb_Cnt_DirDeps}""Th"
#	echo "   Target  Path is ""${TARGT_PATH}"
#	echo "   Current Path is ""${CURNT_PATH}"
	for filename in $(ls  "${TARGT_PATH}") ; do
		if [ -d "${filename}" ];  then
	        #	echo "      Read File name is ""${filename}"
			if [ "${filename}" = "." ] || [ "${filename}" = ".." ] ; then contunue
			elif [ "${GLB_DESTI_PATH}" != "${TARGT_PATH}""/""${filename}" ] ; then
				cd "${filename}"
				CURNT_PATH="$(pwd)"
			#	CURNT_PATH="${TARGT_PATH}""/""${filename}"
				Glb_Cnt_DirDeps=$((Glb_Cnt_DirDeps + Glb_Counter_Value))
	        	#	echo "      changed, Path   : ""${CURNT_PATH}"
			#	echo "      Target  Path    : ""${TARGT_PATH}"
				GoToLastDir "${CURNT_PATH}"
				cd ".."
				CURNT_PATH="$(pwd)"
			#	CURNT_PATH="${TARGT_PATH}"
	        	#	echo "      Rechanged, Path : ""${CURNT_PATH}"
			#	echo "      Target  Path    : ""${TARGT_PATH}"
			else 
			#	echo "      NoChanged, Path : ""${CURNT_PATH}"
				continue
			fi
		fi
	done
	if [ ${Glb_Cnt_DirDeps} -gt 0 ];  then
		DelDupFile "${GLB_TARGT_PATH}" "${CURNT_PATH}"
		Glb_Cnt_DirDeps=$((Glb_Cnt_DirDeps - Glb_Counter_Value))
	fi
	Glb_Cnt_GoToLastDir=$((Glb_Cnt_GoToLastDir - Glb_Counter_Value))
#	echo "   Current Path is ""${CURNT_PATH}"
#	echo "   Target  Path is ""${TARGT_PATH}"
#	echo "   Directory Deps : ""${Glb_Cnt_DirDeps}""Th"
#	echo "   Call Count     : ""${Glb_Cnt_GoToLastDir}""Th"
#	echo "Ended OF FOr Do Loop in GoToLastDir"
#	echo ""
}

echo "=== Start of shell scripty is ""$0"
echo "=== Global target Path is  is ""${GLB_TARGT_PATH}"
echo "=== Destinatination path   is ""${GLB_DESTI_PATH}"
cd "${GLB_TARGT_PATH}"
GoToLastDir "${GLB_TARGT_PATH}"
