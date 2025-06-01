set -x
MkWildCard(){
	local pFile="$1"
	local lFIle=""

	if  [ $(echo ${pFile} | grep '.') ] ; then
		lFile="$(echo ${pFile} | sed 's/\./\*\./1')"
	else
		lFile="${pFile}*"
	fi
	echo "${lFile}"
}

MkWildCard $1
