
# Recompiles the latex not the bib
default: pdf

# Makes everything from scratch
all: pdf bib repeated_pdf clean-log

pdf:
	-pdflatex -interaction nonstopmode master-thesis.tex		# the "-" means --igonre-errors

# Builds the bib
bib:
	-biber master-thesis

repeated_pdf:
	-pdflatex -interaction nonstopmode master-thesis.tex
	-pdflatex -interaction nonstopmode master-thesis.tex

clean-log:
	rm -f *.acn *.aux *.bbl *.bcf *.blg *.glo *.ist *.toc *.lot *.lof *.aux *.log *.out *.run.xml

clean: clean-log
	rm master-thesis.pdf
