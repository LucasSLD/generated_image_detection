#!/bin/bash

pdflatex report.tex
biber report
pdflatex report.tex

latexmk -c report.tex