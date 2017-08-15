#
# makefile for svm_light 
#
# Thorsten Joachims, 2002
#
# Thanks to Ralf Herbrich for the initial version.
#

all:
	@echo ""
	@echo "USAGE make [cygwin|linux|osx]"
	@echo ""

linux:
	make -f Makefile-linux

cygwin:
	make -f Makefile-cygwin

osx:
	make -f Makefile-osx

tidy:
	rm -f *.o 
	rm -f donlp2/*.o
	rm -f donlp2_c/*.o
	rm -f pr_loqo/*.o

clean:	tidy
	rm -f linux-svm_learn*
	rm -f linux-svm_classify*
	rm -f libsvmlight.*
	rm -f svmlight.dll
	rm -f win-svm_learn*
	rm -f win-svm_classify*
	rm -f *.o

