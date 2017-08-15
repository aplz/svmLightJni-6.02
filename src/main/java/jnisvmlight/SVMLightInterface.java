/*
 * JNI_SVM-light - A Java Native Interface for SVM-light
 * 
 * Copyright (C) 2005 
 * Tom Crecelius & Martin Theobald 
 * Max-Planck Institute for Computer Science
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */

package jnisvmlight;

import com.google.common.collect.Lists;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

/**
 * The main interface class that transfers the training data to the SVM-light library by a native call. Optionally takes as input an
 * individually modified set of training parameters or an array of string parameters that exactly simulate the command line input parameters
 * used by the SVM-light binaries. This class can also be used for native classification calls.
 *
 * @author Tom Crecelius & Martin Theobald, including a bug fix by George Shaw (MIT)

 * @author Anja Pilz
 */
public class SVMLightInterface implements Serializable {
    private static final Logger LOGGER = LoggerFactory.getLogger(SVMLightInterface.class);
    /**
	 * Apply an in-place quicksort prior to each native training call to SVM-light. SVM-light requires each input feature vector to be
     * sorted in ascending order of dimensions. Disable this option if you are sure to provide sorted vectors already.
     */
    public static boolean SORT_INPUT_VECTORS = true;

    static {
        // TODO: this not good practice. Either the client code should issue this statement accordingly or
        // the path must be set more elegantly.
        System.load(System.getProperty("user.dir") + "/lib/svmlight-64.dll");
    }

    /**
     * Reads a set of labeled training vectors from a URL. The format is compatible to the SVM-light training files.
     *
     * @param file
     * @param numOfLinesToSkip
     *
     * @return
     *
     * @throws ParseException
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static LabeledFeatureVector[] fromPath(Path file, int numOfLinesToSkip) throws ParseException {

        LOGGER.info("Reading ... " + file);
        ArrayList data = new ArrayList();
        LabeledFeatureVector[] trainingData = null;

        try {
            List<String> lines = Files.readAllLines(file);
            int cnt = 0;
            for (String line : lines) {
                if (line.startsWith("#")) {
                    // Skip lines indicated as a comment.
                    continue;
                }
                cnt++;
                if (cnt <= numOfLinesToSkip) {
                    continue;
                }
                String tokens[] = line.trim().split("[ \\t\\n\\x0B\\f\\r\\[\\]]");
                int queryId = 0;
                if (tokens.length > 2) {
                    String label = tokens[0];
                    // System.out.println("Label: " + label);
                    // TODO: was ist factor?
                    String factor = "";
                    if (tokens[1].startsWith("qid")) {
                        queryId = Integer.parseInt(tokens[1].substring(tokens[1].indexOf(":") + 1, tokens[1].length()));
                        // hack: factor is set to 1.0 when reading data like this
                        // especially when reading the ranking svm training data, the parser does currently not support
                        // factors
                        factor = "1.0";
                    } else {
                        factor = tokens[1].substring(0, tokens[1].length() - 1);
                    }

                    List<String> dimensionsList = Lists.newArrayList();
                    List<String> valuesList = Lists.newArrayList();
                    for (int tokenCounter = 2; tokenCounter < tokens.length; tokenCounter++) {
                        String dimensionValue = tokens[tokenCounter];
                        if (dimensionValue.trim().startsWith("#")) {
                            break;
                        }

                        int idx = dimensionValue.indexOf(':');
                        if (idx >= 0) {
                            dimensionsList.add(dimensionValue.substring(0, idx));
                            valuesList.add(dimensionValue.substring(idx + 1, dimensionValue.length()));
                        } else {
                            throw new ParseException("Parse error in FeatureVector of file '" + file.toString()
                                    + "' at line: " + cnt + ", token: " + tokenCounter
                                    + ". Could not estimate a \"int:double\" pair ?! " + file.toString()
                                    + " contains a wrongly defined feature vector!", 0);
                        }
                    }
                    if (dimensionsList.size() > 0) {
                        double labelValue = Double.parseDouble(label);
                        double factorValue = Double.parseDouble(factor);
                        int[] dimensions = dimensionsList.stream().mapToInt(Integer::parseInt).toArray();
                        double[] values = valuesList.stream().mapToDouble(Double::parseDouble).toArray();
                        LabeledFeatureVector labeledFeatureVector = new LabeledFeatureVector(labelValue, dimensions, values);
                        labeledFeatureVector.setFactor(factorValue);
                        labeledFeatureVector.setQid(queryId);
                        data.add(labeledFeatureVector);
                    }
                } else {
                    throw new ParseException("Parse error in FeatureVector of file '" + file.toString() + "' at line: "
                            + cnt + ". " + " Wrong format of the labeled feature vector?", 0);
                }
            }
            if (data.size() > 0) {
                data.toArray(new LabeledFeatureVector[data.size()]);
                trainingData = new LabeledFeatureVector[data.size()];
                for (int i = 0; i < data.size(); i++) {
                    trainingData[i] = (LabeledFeatureVector) data.get(i);
                }
            } else {
                throw new ParseException("No labeled features found within " + cnt + "lines of file '"
                        + file.toString() + "'.", 0);
            }
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }
        return trainingData;
    }

    /**
     * Internal usage...
     */
    protected TrainingParameters m_tp;

    /**
     * Performs a classification step as a native call to SVM-light. If this method is used exclusively, no additional SVMLightModel object
     * has to be kept in the Java runtime process.
     */
    public native double classifyNative(FeatureVector doc);

    public TrainingParameters getTrainingParameters() {
        return m_tp;
    }

    private void quicksort(int[] dims, double[] vals, int low, int high) {
		if (low >= high) {
			return;
		}

        int leftIdx = low;
        int pivot = low;
        int rightIdx = high;
        pivot = (low + high) / 2;
        while (leftIdx <= pivot && rightIdx >= pivot) {
			while (dims[leftIdx] < dims[pivot] && leftIdx <= pivot) {
				leftIdx++;
			}
			while (dims[rightIdx] > dims[pivot] && rightIdx >= pivot) {
				rightIdx--;
			}
            int tmp = dims[leftIdx];
            dims[leftIdx] = dims[rightIdx];
            dims[rightIdx] = tmp;
            double tmpd = vals[leftIdx];
            vals[leftIdx] = vals[rightIdx];
            vals[rightIdx] = tmpd;
            leftIdx++;
            rightIdx--;
			if (leftIdx - 1 == pivot) {
				pivot = rightIdx = rightIdx + 1;
			} else if (rightIdx + 1 == pivot) {
				pivot = leftIdx = leftIdx - 1;
			}
            quicksort(dims, vals, low, pivot - 1);
            quicksort(dims, vals, pivot + 1, high);
        }
    }

    private void sort(FeatureVector[] trainingData) {
        for (int i = 0; i < trainingData.length; i++) {
            if (trainingData[i] != null) {
                quicksort(trainingData[i].m_dims, trainingData[i].m_vals, 0, trainingData[i].size() - 1);
                // verifyIsSorted(trainingData[i].m_dims);
            }
        }
    }

    private native SVMLightModel trainmodel(LabeledFeatureVector[] traindata, TrainingParameters p);

    public SVMLightModel trainModel(LabeledFeatureVector[] trainingData) {
        this.m_tp = new TrainingParameters();
        return this.trainmodel(trainingData, this.m_tp);
    }

    public SVMLightModel trainModel(LabeledFeatureVector[] trainingData, String[] argv) {
        this.m_tp = new TrainingParameters(argv);
        return this.trainmodel(trainingData, this.m_tp);
    }

    public SVMLightModel trainModel(LabeledFeatureVector[] trainingData, TrainingParameters tp) {
        this.m_tp = tp;
        if (SORT_INPUT_VECTORS) {
            sort(trainingData);
        }
        return trainmodel(trainingData, m_tp);
    }

}
