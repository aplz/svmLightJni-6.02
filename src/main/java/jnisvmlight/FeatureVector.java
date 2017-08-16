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

import java.util.stream.DoubleStream;

/**
 * A feature vector. Features are dimension-value pairs. This class implements a simple dictionary data structure to map dimensions onto
 * their values. Note that for convenience, features do not have be sorted according to their dimensions at this point. The SVMLightTrainer
 * class has an option for sorting input vectors prior to training.
 *
 * @author Tom Crecelius & Martin Theobald
 * @author Anja Pilz
 */
public class FeatureVector implements java.io.Serializable {

    protected int[] m_dims;
    protected double m_factor;
    protected double[] m_vals;

    /**
     * Instantiate a feature vector with a given {@code factor}, {@code dimensions} and {@code values}.
     *
     * @param factor     the feature vectors factor.
     * @param dimensions the feature dimensions.
     * @param values     the feature values.
     */
    public FeatureVector(double factor, int[] dimensions, double[] values) {
        this.m_factor = factor;
        this.m_dims = dimensions;
        this.m_vals = values;
    }

    public FeatureVector(int size) {
        this(1.0, new int[size], new double[size]);
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
     * Returns the cosine similarity between two feature vectors.
     */
    public double getCosine(FeatureVector v) {
        double cosine = 0.0;
        int dim;
        double q_i, d_i;
        for (int i = 0; i < Math.min(this.size(), v.size()); i++) {
            dim = v.getDimAt(i);
            q_i = v.getValueAt(dim);
            d_i = this.getValueAt(dim);
            cosine += q_i * d_i;
        }
        return cosine / (this.getL2Norm() * v.getL2Norm());
    }

    public int getDimAt(int index) {
        return m_dims[index];
    }

    public double getFactor() {
        return m_factor;
    }

    /**
     * Returns the linear norm factor of this vector's values (i.e., the sum of it's values).
     */
    public double getL1Norm() {
        return DoubleStream.of(m_vals).sum();
    }

    /**
     * Returns the L2 norm factor of this vector's values.
     */
    public double getL2Norm() {
        double square_sum = 0.0;
        for (int i = 0; i < m_vals.length; i++) {
            square_sum += (m_vals[i] * m_vals[i]);
        }
        return Math.sqrt(square_sum);
    }

    public double getValueAt(int index) {
        return m_vals[index];
    }

    /**
     * Performs a linear normalization to the value 1.
     */
    public void normalizeL1() {
        normalizeL1(getL1Norm());
    }

    /**
     * Performs a linear normalization to the given norm value.
     */
    public void normalizeL1(double norm) {
        for (int i = 0; i < m_vals.length; i++) {
            if (m_vals[i] > 0) {
                m_vals[i] /= norm;
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

    public void setFactor(double factor) {
        this.m_factor = factor;
    }

    public void setFeatures(int[] dimensions, double[] values) {
        this.m_dims = dimensions;
        this.m_vals = values;
    }

    public int size() {
        return m_dims.length;
    }

    public String toString() {
        String s = "";
        for (int i = 0; i < m_vals.length; i++) {
            s += "" + m_dims[i] + ":" + m_vals[i] + "" + (i < m_vals.length - 1 ? " " : "");
        }
        return s;
    }
}
