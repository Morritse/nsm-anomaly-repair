TEX = paper.tex
CANONICAL_PDF = paper.pdf
SUBMISSION_PDF = nsm-anomaly-repair.pdf

.PHONY: all clean

all: $(SUBMISSION_PDF)

$(CANONICAL_PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

$(SUBMISSION_PDF): $(CANONICAL_PDF)
	cp $(CANONICAL_PDF) $(SUBMISSION_PDF)

clean:
	rm -f paper.aux paper.log paper.out
