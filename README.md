# svm light JNI

This project is a Java Native Interface for Thorsten Joachims' [svm_light-6.02](https://www.cs.cornell.edu/People/tj/svm_light/). It is based upon Tom Crecelius' and Martin Theobald's [original version](http://people.mpi-inf.mpg.de/~mtb/svmlight/) but provides a fix that enables
the training of Ranking SVMs. 
In the original version, the parameter _query id_ was neither propagated nor imported from training data.
This has been fixed and tested [here](https://github.com/aplz/svmLightJni-6.02/blob/master/src/test/java/jnisvmlight/SVMLightInterfaceTest.java).

## Resources
Apart from the Java code, this repository includes the following resources:

* the original SVM light implementation by Thorsten Joachims (see lib/svmlight-6.02)
* the interface code and Makefiles to build libraries for Linux, Windows and OS (see lib/svmlight-6.02)
* a pre-compiled 64bit dll tested on Windows 10 with Java 8   
* a training and test data set for Ranking SVMs (see data/example3). The data in this folder has been downloaded from the above svm light website and is included here to enable interface testing.

The dll can be build with cygwin through
```
    make cygwin tidy
```
Note: for Windows using the mingw compiler, the path in the Makefile might need to be adapted (as well as the Java path).


## References
* Ranking SVMs: T. Joachims, Optimizing Search Engines Using Clickthrough Data, Proceedings of the ACM Conference on Knowledge Discovery and Data Mining (KDD), ACM, 2002
* SVM Light: https://www.cs.cornell.edu/People/tj/svm_light/
* Original JNI: http://people.mpi-inf.mpg.de/~mtb/svmlight/

## Licenses
SVM Light is subject to Thorsten Joachim's [license](https://github.com/aplz/svmLightJni-6.02/blob/master/lib/svmlight-6.02/LICENSE.txt).

