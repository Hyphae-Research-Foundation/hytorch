PYTHON ?= python3
TECTONIC ?= tectonic
SOURCE_DATE_EPOCH ?= 1789171200
EN_SOURCE_DATE_EPOCH ?= 1789257600

.PHONY: all verify figures figures-en pdf pdf-en preview preview-en bilingual clean

all: pdf

verify:
	$(PYTHON) -B data/verify_selection.py
	$(PYTHON) -B design/verify_design.py

figures:
	$(PYTHON) -B make_figures.py

figures-en:
	$(PYTHON) -B make_figures.py --language en

pdf: verify figures
	mkdir -p build
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) --keep-logs --outdir build report.tex
	cp build/report.pdf report.pdf

preview: pdf
	pdftoppm -f 1 -singlefile -scale-to 1400 -png report.pdf preview

pdf-en: verify figures-en
	mkdir -p build
	SOURCE_DATE_EPOCH=$(EN_SOURCE_DATE_EPOCH) $(TECTONIC) --keep-logs --outdir build report-en.tex
	cp build/report-en.pdf report-en.pdf

preview-en: pdf-en
	pdftoppm -f 1 -singlefile -scale-to 1400 -png report-en.pdf preview-en

bilingual: preview preview-en

clean:
	rm -rf build
