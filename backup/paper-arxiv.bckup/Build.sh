#!/bin/sh

# Clean auxiliary files that can cause hyperref instability
rm -f $1.out

# Build: pdflatex -> bibtex -> pdflatex x3
# The extra passes resolve citations, cross-refs, and hyperref outlines.
pdflatex -interaction=nonstopmode $1.tex
bibtex $1
pdflatex -interaction=nonstopmode $1.tex
pdflatex -interaction=nonstopmode $1.tex
pdflatex -interaction=nonstopmode $1.tex
