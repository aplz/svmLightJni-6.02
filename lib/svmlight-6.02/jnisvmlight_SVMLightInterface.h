/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class jnisvmlight_SVMLightInterface */

#ifndef _Included_jnisvmlight_SVMLightInterface
#define _Included_jnisvmlight_SVMLightInterface
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     jnisvmlight_SVMLightInterface
 * Method:    classifyNative
 * Signature: (Ljnisvmlight/FeatureVector;)D
 */
JNIEXPORT jdouble JNICALL Java_jnisvmlight_SVMLightInterface_classifyNative
  (JNIEnv *, jobject, jobject);

/*
 * Class:     jnisvmlight_SVMLightInterface
 * Method:    trainmodel
 * Signature: ([Ljnisvmlight/LabeledFeatureVector;Ljnisvmlight/TrainingParameters;)Ljnisvmlight/SVMLightModel;
 */
JNIEXPORT jobject JNICALL Java_jnisvmlight_SVMLightInterface_trainmodel
  (JNIEnv *, jobject, jobjectArray, jobject);

#ifdef __cplusplus
}
#endif
#endif
