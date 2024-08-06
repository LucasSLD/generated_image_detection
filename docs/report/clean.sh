#!/bin/bash

latexmk report.tex -c

rm report.synctex.gz || true

BASENAME="presentation"

# Define the extensions you want to protect
PROTECT_EXTENSIONS=("tex" "pdf")

# Iterate over all files that start with the BASENAME
for file in ${BASENAME}.*; do
    # Get the file extension
    ext="${file##*.}"
    
    # Check if the file's extension is not in the protected list
    if [[ ! " ${PROTECT_EXTENSIONS[@]} " =~ " ${ext} " ]]; then
        # If not protected, remove the file
        rm "$file"
    fi
done