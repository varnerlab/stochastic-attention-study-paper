# Makefile for paper-arxiv build
#
# Usage:
#   make            -- build Paper_v1.pdf (default)
#   make pdf        -- same as above
#   make clean      -- remove auxiliary files (keep PDF)
#   make distclean  -- remove auxiliary files and PDF
#   make rebuild    -- distclean + pdf

PAPER    = Paper_v1
TEX      = $(PAPER).tex
PDF      = $(PAPER).pdf
BIB      = References_v1.bib
SECTIONS = $(wildcard sections/*.tex)

LATEX    = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX   = bibtex

AUX_EXTS = aux bbl blg log out toc lof lot fls fdb_latexmk synctex.gz nav snm vrb

.PHONY: all pdf clean distclean rebuild

all: pdf

pdf: $(PDF)

$(PDF): $(TEX) $(BIB) $(SECTIONS)
	rm -f $(PAPER).out
	$(LATEX) $(TEX)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

clean:
	$(foreach ext,$(AUX_EXTS),rm -f $(PAPER).$(ext);)

distclean: clean
	rm -f $(PDF)

rebuild: distclean pdf
