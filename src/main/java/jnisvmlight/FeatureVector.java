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

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;

import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * A feature vector. Features are dimension-value pairs. This class implements a simple dictionary data structure to map dimensions onto
 * their values. Note that for convenience, features do not have be sorted according to their dimensions at this point. The SVMLightTrainer
 * class has an option for sorting input vectors prior to training.
 *
 * @author Tom Crecelius & Martin Theobald
 * @author Anja Pilz
 */
public class FeatureVector implements java.io.Serializable {
    // field names must not be changed!
    protected int[] m_dims;
    protected double m_factor;
    protected double[] m_vals;

    /**
     * Instantiate a feature vector with a given {@code factor}, {@code dimensions} and {@code values}.
     *
     * @param factor     the feature vector's factor.
     * @param dimensions the feature dimensions.
     * @param values     the feature values.
     */
    public FeatureVector(double factor, int[] dimensions, double[] values) {
        Preconditions.checkArgument(dimensions.length == values.length, "The number of dimensions and values must be the same!");
        Preconditions.checkArgument(IntStream.of(dimensions).min().getAsInt() > 0, "Dimensions must start at 1!");
        this.m_factor = factor;
        this.m_dims = dimensions;
        this.m_vals = values;
    }

    /**
     * Instantiate a feature vector with given {@code dimensions} and {@code values} and a default {@link #m_factor} of 1.0.
     *
     * @param dimensions the feature dimensions.
     * @param values     the feature values.
     */
    public FeatureVector(int[] dimensions, double[] values) {
        this(1.0, dimensions, values);
    }

    /**
     * This constructor must not be deleted. It is through {@link LabeledFeatureVector} required in the JNI at the GetMethodID step (lines
     * 46 to 50 in svm_jni .c).
     */
    FeatureVector() {
    }

    /**
     * Returns the cosine similarity between two feature vectors.
     *
     * @param other the second feature vector.
     * @return the cosine similarity between two feature vectors.
     */
    public double getCosine(FeatureVector other) {
        double cosine = 0.0;
        Map<Integer, Double> tempThis = Maps.newHashMap();
        // temporarily store the vector in a map so that we can more easily check for common dimensions
        for (int i = 0; i < m_dims.length; i++) {
            tempThis.put(m_dims[i], m_vals[i]);
        }
        for (int i = 0; i < other.m_dims.length; i++) {
            if (tempThis.containsKey(other.m_dims[i])) {
                cosine += tempThis.get(other.m_dims[i]) * other.m_vals[i];
            }
        }
        return cosine / (this.getL2Norm() * other.getL2Norm());
    }

    /**
     * Returns this vector's factor.
     *
     * @return this vector's factor.
     */
    public double getFactor() {
        return m_factor;
    }

    /**
     * Set this vector's factor.
     *
     * @param factor the factor to set.
     */
    public void setFactor(double factor) {
        this.m_factor = factor;
    }

    /**
     * Returns the linear norm factor of this vector's values (i.e., the sum of it's values).
     *
     * @return the linear norm factor of this vector's values.
     */
    public double getL1Norm() {
        return DoubleStream.of(m_vals).sum();
    }

    /**
     * Returns the L2 norm factor of this vector's values.
     *
     * @return the L2 norm factor of this vector's values.
     */
    public double getL2Norm() {
        double square_sum = 0.0;
        for (int i = 0; i < m_vals.length; i++) {
            square_sum += (m_vals[i] * m_vals[i]);
        }
        return Math.sqrt(square_sum);
    }


    /**
     * Performs a linear normalization to the value 1.
     */
    public void normalizeL1() {
        double l1Norm = getL1Norm();
        for (int i = 0; i < m_vals.length; i++) {
            if (m_vals[i] > 0) {
                m_vals[i] /= l1Norm;
            }
        }
    }

    /**
     * Performs an L2 normalization to the value 1.
     */
    public void normalizeL2() {
        double norm = Math.pow(getL2Norm(), 2);
        for (int i = 0; i < m_vals.length; i++) {
            m_vals[i] = Math.pow(m_vals[i], 2) / norm;
        }
    }

    /**
     * Returns the number of features in this vector, i.e. the number of set dimensions.
     *
     * @return the number of features in this vector.
     */
    public int size() {
        return m_dims.length;
    }

    @Override
    public String toString() {
        String s = "";
        for (int i = 0; i < m_vals.length; i++) {
            s += "" + m_dims[i] + ":" + m_vals[i] + "" + (i < m_vals.length - 1 ? " " : "");
        }
        return s;
    }
}
