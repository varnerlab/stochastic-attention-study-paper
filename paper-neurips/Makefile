# Makefile for the NeurIPS submission
# ------------------------------------
# Default:        make           → build Paper_v1.pdf
# Force rebuild:  make rebuild   → wipe aux + rebuild
# Clean aux:      make clean     → keep PDF, remove all aux artifacts
# Wipe all:       make distclean → remove aux artifacts AND the PDF
# Open PDF:       make view      → open the PDF in the default viewer
# Verify build:   make verify    → fail on undefined refs / citations / overfull hboxes
# Page audit:     make pages     → print the section→page TOC mapping
# Build skill:    make help      → list targets

DOC      := Paper_v1
TEX      := $(DOC).tex
BIB      := References_v1.bib
SECTIONS := $(wildcard sections/*.tex)
FIGS     := $(wildcard figs/*.png) $(wildcard figs/*.pdf) $(wildcard figs/*.jpg)
STY      := neurips_2026.sty

PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX   := bibtex

# Aux files produced by a build (keep this list synced with .gitignore).
AUX := $(DOC).aux $(DOC).log $(DOC).bbl $(DOC).blg $(DOC).out $(DOC).toc \
       $(DOC).lof $(DOC).lot $(DOC).loa $(DOC).spl $(DOC).fls \
       $(DOC).fdb_latexmk $(DOC).synctex.gz

.PHONY: all rebuild clean distclean view verify pages help

all: $(DOC).pdf

# Standard pdflatex → bibtex → pdflatex × 3 dance for natbib + hyperref.
# Depend on every .tex section, the .bib, the .sty, and every figure so any
# of them touched triggers a rebuild.
$(DOC).pdf: $(TEX) $(SECTIONS) $(BIB) $(STY) $(FIGS)
	rm -f $(DOC).out
	$(PDFLATEX) $(TEX)
	$(BIBTEX)   $(DOC)
	$(PDFLATEX) $(TEX)
	$(PDFLATEX) $(TEX)
	$(PDFLATEX) $(TEX)

# `rebuild` must clobber the PDF too, otherwise `all` sees the stale PDF as
# up-to-date and the build silently no-ops.
rebuild: distclean all

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(DOC).pdf

view: $(DOC).pdf
	@case "$$(uname -s)" in \
	  Darwin) open $(DOC).pdf ;; \
	  Linux)  xdg-open $(DOC).pdf ;; \
	  *)      echo "Open $(DOC).pdf manually" ;; \
	esac

# Fail loudly if the build introduced undefined refs/citations or overfull
# hboxes. Useful in CI or as a pre-commit gate. If aux/log are missing
# (e.g. after `make clean`), force a fresh build so the diagnostics aren't
# misleadingly empty.
verify: $(DOC).pdf
	@[ -f $(DOC).log ] || $(MAKE) rebuild >/dev/null
	@echo "── undefined references ──"
	@! grep -E "(LaTeX Warning: Reference|LaTeX Warning: Citation|undefined)" $(DOC).log | grep -v "^pdfTeX warning" || (echo "FAIL: undefined refs/citations" && exit 1)
	@echo "── overfull hboxes ──"
	@! grep -E "Overfull \\\\hbox" $(DOC).log || (echo "WARN: overfull hboxes present" && exit 0)
	@echo "── multiply-defined labels ──"
	@! grep -E "multiply defined" $(DOC).log || (echo "FAIL: multiply-defined labels" && exit 1)
	@echo "OK"

# Print the section → page mapping from the .aux so you can eyeball whether
# the main text still fits in the 9-page NeurIPS limit.
pages: $(DOC).pdf
	@[ -f $(DOC).aux ] || $(MAKE) rebuild >/dev/null
	@grep -F "writefile" $(DOC).aux | grep -F "{toc}" | \
	  sed -E 's/.*contentsline \{(section|subsection|paragraph)\}\{[^{]*\\numberline \{([^}]*)\}([^}]*)\}\{([0-9]+)\}.*/  \1 \2: \3 (page \4)/; s/.*contentsline \{paragraph\}\{([^}]*)\}\{([0-9]+)\}.*/  paragraph: \1 (page \2)/'

help:
	@echo "Targets:"
	@echo "  make            build $(DOC).pdf if any source changed"
	@echo "  make rebuild    clean aux + full rebuild"
	@echo "  make clean      remove aux files (keeps PDF)"
	@echo "  make distclean  remove aux files AND the PDF"
	@echo "  make view       open the PDF"
	@echo "  make verify     fail on undefined refs/cites or overfull hboxes"
	@echo "  make pages      print section → page mapping for the 9-page audit"
