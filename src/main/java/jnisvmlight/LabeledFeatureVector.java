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

/**
 * A labeled feature vector.
 *
 * @author Tom Crecelius & Martin Theobald
 * @author Anja Pilz
 */
public class LabeledFeatureVector extends FeatureVector implements java.io.Serializable {

    protected double m_label;
    protected int m_qid;

    /**
     * Instantiate a labeled feature vector with the given {@code label}, {@code dimensions} and {@code values}.
     *
     * @param label      the feature vector's label.
     * @param dimensions the feature dimensions.
     * @param values     the feature values.
     */
    public LabeledFeatureVector(double label, int[] dimensions, double[] values) {
        super(dimensions, values);
        this.m_label = label;
    }

    /**
     * Returns the feature vector's label.
     *
     * @return the feature vector's label.
     */
    public double getLabel() {
        return m_label;
    }

    @Override
    public String toString() {
        return (m_label * m_factor) + " " + super.toString() + "\n";
    }

    /**
     * Returns the feature vector's query ID.
     *
     * @return the feature vector's query ID.
     */
    public int getQueryId() {
        return this.m_qid;
    }

    /**
     * Set the feature vector's query ID.
     *
     * @param m_qid the query ID to set.
     */
    public void setQueryId(int m_qid) {
        this.m_qid = m_qid;
    }
}
