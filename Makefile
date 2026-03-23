# Main tex file (without .tex extension)
DOC = Paper_v1

ARXIV_DIR = paper-arxiv
NEURIPS_DIR = paper-neurips

LATEXMK = pdflatex -interaction=nonstopmode

AUX_EXTS = aux bbl blg log out fls fdb_latexmk synctex.gz

# ---------- targets ----------

.PHONY: all arxiv neurips clean clean-arxiv clean-neurips

all: arxiv neurips

arxiv:
	cd $(ARXIV_DIR) && $(LATEXMK) $(DOC).tex && bibtex $(DOC) && $(LATEXMK) $(DOC).tex && $(LATEXMK) $(DOC).tex

neurips:
	cd $(NEURIPS_DIR) && $(LATEXMK) $(DOC).tex && bibtex $(DOC) && $(LATEXMK) $(DOC).tex && $(LATEXMK) $(DOC).tex

clean: clean-arxiv clean-neurips

clean-arxiv:
	cd $(ARXIV_DIR) && rm -f $(foreach ext,$(AUX_EXTS),$(DOC).$(ext))

clean-neurips:
	cd $(NEURIPS_DIR) && rm -f $(foreach ext,$(AUX_EXTS),$(DOC).$(ext))
