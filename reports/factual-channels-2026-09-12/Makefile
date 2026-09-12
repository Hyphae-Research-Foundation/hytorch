PYTHON ?= python3
TECTONIC ?= tectonic
SOURCE_DATE_EPOCH ?= 1789171200

.PHONY: all verify figures pdf preview clean

all: pdf

verify:
	$(PYTHON) -B data/verify_selection.py
	$(PYTHON) -B design/verify_design.py

figures:
	$(PYTHON) -B make_figures.py

pdf: verify figures
	mkdir -p build
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) --keep-logs --outdir build report.tex
	cp build/report.pdf report.pdf

preview: pdf
	pdftoppm -f 1 -singlefile -scale-to 1400 -png report.pdf preview

clean:
	rm -rf build
