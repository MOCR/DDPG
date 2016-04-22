File = ddpg

all:
	cd ./images-src/ ; make
	pdflatex ${File}.tex
	bibtex ${File}.aux
	pdflatex ${File}.tex
	pdflatex ${File}.tex
