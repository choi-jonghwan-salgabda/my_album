
#!/bin/bash

SHELL_ARG_NUM=$#
GLB_SHELL_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
GLB_TARGT_PATH="$1"
GLB_DESTI_PATH="$2"
GLB_CURNT_PATH= pwd
Glb_Cnt_GoToLastDir=0
Glb_Cnt_DelDupFile=0
Glb_Counter_Value=1
Glb_Cnt_DirDeps=0


GoToLastDir(){
	TARGT_PATH="$1"
	CURNT_PATH="$2"

 	echo ""
	Glb_Cnt_GoToLastDir=$((Glb_Cnt_GoToLastDir + Glb_Counter_Value))
 	echo "Start OF FOr Do Loop in GoToLastDir"
 	echo "Start Call Count     : ""${Glb_Cnt_GoToLastDir}""Th"
 	echo "Start Directory Deps : ""${Glb_Cnt_DirDeps}""Th"
 	echo "Start GlbTargtrath   : ""${GLB_TARGT_PATH}"
 	echo "Start Target  Path   : ""${TARGT_PATH}"
 	echo "Start Current Path   : ""${CURNT_PATH}"
	for filename in $( ls ) ; do
#		if [ -f "${filename}" ] && [ "${TARGT_PATH}" != "${CURNT_PATH}" ] ; then
 		if [ -f "${filename}" ] ; then
         		echo "      Read file Name is ""${filename}"
		 	find "${GLB_TARGT_PATH}" \! \( \( -type d -path 'lost+found' -o -type d -path './.Trash-1000/*' -o -type d -path "${CURNT_PATH}" \) -prune \) -type f -iname "${filename}" -print
		 #	find "${GLB_TARGT_PATH}" \! \( \( -type d -path 'lost+found' -o -type d -path './.Trash-1000/*' -o -type d -path "${TARGT_PATH}" \) -prune \) -type f -iname "${filename}" -exec tar --remove-files -uf "${GLB_DESTI_PATH}""/""${filename}".tar {} +;
		elif [ -d "${filename}" ]; then
			if [ "${filename}" == "." ] || [ "${filename}" == ".." ] ; then contunue
			else
# 				if [ "${TARGT_PATH}" = "./" ]; then
# 					CURNT_PATH="${filename}"
# 				elif [ "${TARGT_PATH:${#TARGT_PATH} -1:1}" = "/" ]; then
  				if [ "${TARGT_PATH:${#TARGT_PATH} -1:1}" = "/" ]; then
					CURNT_PATH="${TARGT_PATH}""${filename}"
				else
					CURNT_PATH="${TARGT_PATH}""/""${filename}"
				fi
#				echo "      Maked Current path     : ""${CURNT_PATH}"
# 			 	echo "      Maked Target  Path     : ""${TARGT_PATH}"
 				if [ "${GLB_DESTI_PATH}" != "${CURNT_PATH}" ] ; then
#					echo "      Change Current pwd     : ""$(pwd)"
#					echo "      Change Current path    : ""${CURNT_PATH}"
#			 		echo "      Change Target  Path    : ""${TARGT_PATH}"
					cd "${CURNT_PATH}"
#					echo "      Changed Current pwd    : ""$(pwd)"
#					echo "      Changed Current path   : ""${CURNT_PATH}"
# 			 		echo "      Changed Target  Path   : ""${TARGT_PATH}"
					Glb_Cnt_DirDeps=$((Glb_Cnt_DirDeps + Glb_Counter_Value))
 					GoToLastDir "${TARGT_PATH}" "${CURNT_PATH}"
					cd ".."
					Glb_Cnt_DirDeps=$((Glb_Cnt_DirDeps - Glb_Counter_Value))
 			 		CURNT_PATH="${TARGT_PATH}"
#					echo "      Rechanged Current pwd  : ""$(pwd)"
#					echo "      Rechanged Current path : ""${CURNT_PATH}"
# 			 		echo "      Rechanged Target  Path : ""${TARGT_PATH}"
 				else 
# 			 		echo "      NoChanged, Path        : ""${CURNT_PATH}"
 					continue
 				fi
			fi
		fi
	done
 	echo "Ended Current Path   : ""${CURNT_PATH}"
 	echo "Ended Target  Path   : ""${TARGT_PATH}"
 	echo "Ended Directory Deps : ""${Glb_Cnt_DirDeps}""Th"
 	echo "Ended Call Count     : ""${Glb_Cnt_GoToLastDir}""Th"
 	echo "Ended OF FOr Do Loop in GoToLastDir"
 	echo ""
}

echo "=== Start shell scripte is ""$0"
echo "=== target Path is  is ""${GLB_TARGT_PATH}"
echo "=== Destinatin path is ""${GLB_DESTI_PATH}"
cd "${GLB_TARGT_PATH}"
GoToLastDir "${GLB_TARGT_PATH}" "${GLB_TARGT_PATH}"
