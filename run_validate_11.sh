#!/bin/bash

source ./scripts/utils.sh

# Check version of OS
check_os

# Install the necessary packages to run validation
check_packages

# Check that required folders exist 
check_folders

# Compile and build implementation library against
# validation test driver
scripts/compile_and_link.sh
retcode=$?
if [[ $retcode != 0 ]]; then
	exit $failure
fi

# Set dynamic library path to the folder location of the developer's submission library
export LD_LIBRARY_PATH=$(pwd)/lib

# Run testdriver against linked library
# and validation images
scripts/run_testdriver.sh
retcode=$?
if [[ $retcode != 0 ]]; then
	exit $failure
fi

outputDir="validation"
# Do some sanity checks against the output logs
echo -n "Sanity checking validation output "
for input in enroll verif match match_multiperson
do
	numInputLines=$(cat input/$input.txt | wc -l)
	numLogLines=$(sed '1d' $outputDir/$input.log | wc -l)
	if [ "$numInputLines" != "$numLogLines" ]; then
		echo "[ERROR] The $outputDir/$input.log file has the wrong number of lines.  It should contain $numInputLines but contains $numLogLines.  Please re-run the validation test."
		exit $failure
	fi

	# Check return codes
	numFail=$(sed '1d' $outputDir/$input.log | awk '{ if($4!=0) print }' | wc -l)
	if [ "$numFail" != "0" ]; then
		echo -e "\n${bold}[WARNING] The following entries in $input.log generated non-successful return codes:${normal}"
		sed '1d' $outputDir/$input.log | awk '{ if($4!=0) print }'
	fi

	if [ "$input" == "match" ]; then
		# Check that at least 50% of match scores are unique
		minUniqScores=$(echo "$numInputLines * 0.5" | bc | awk '{printf("%d\n",$1 + 0.5)}')
		numUniqScores=$(sed '1d' $outputDir/$input.log | awk '{ print $3 }' | uniq | wc -l)
		if [ "$numUniqScores" -lt "$minUniqScores" ]; then
			echo -e "\n${bold}[WARNING] Your software produces $numUniqScores unique match scores, which is less than 50% unique match score values.  In order to conduct useful analysis, we highly recommend that you fix your software such that it generates at least 50% unique match scores on this validation set.${normal}"
		fi
	
		# Check for negative scores coming from the algorithm
		numNegativeScores=$(sed '1d' $outputDir/$input.log | awk '{ if($4==0 && ($3+0)<0) print }' | wc -l)
		if [ "$numNegativeScores" -gt "0" ]; then
			echo -e "\n${bold}[ERROR] Your software produces $numNegativeScores negative match scores.  Negative scores are not allowed, per the FRVT General Specifications Document.  Please fix your software.${normal}"
			exit $failure
			
		fi
	fi	
done
echo "[SUCCESS]"

# Create submission archive
echo -n "Creating submission package "
libstring=$(basename `ls ./lib/libfrvt_11_*_???.so`)
libstring=${libstring%.so}

for directory in config lib validation doc
do
	if [ ! -d "$directory" ]; then
		echo "[ERROR] Could not create submission package.  The $directory directory is missing."
		exit $failure	
	fi
done

# write OS to text file
log_os
# append frvt_structs.h version to submission filename
version=$(get_frvt_header_version)

libstring="$libstring.v${version}"
tar -zcf $libstring.tar.gz ./doc ./config ./lib ./validation
echo "[SUCCESS]"
echo "
#################################################################################################################
A submission package has been generated named $libstring.tar.gz.  DO NOT RENAME THIS FILE. 

This archive must be properly encrypted and signed before transmission to NIST.
This must be done according to these instructions - https://www.nist.gov/sites/default/files/nist_encryption.pdf
using the LATEST FRVT Ongoing public key linked from -
https://www.nist.gov/itl/iad/image-group/products-and-services/encrypting-softwaredata-transmission-nist.

For example:
      gpg --default-key <ParticipantEmail> --output <filename>.gpg \\\\
      --encrypt --recipient frvt@nist.gov --sign \\\\
      libfrvt_11_<organization>_<three-digit submission sequence>.v<validation_package_version>.tar.gz

Submit the encrypted file and your public key to NIST following the instructions at http://pages.nist.gov/frvt/html/frvt_submission_form.html.
##################################################################################################################
"
