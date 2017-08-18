/**
 *
 */
package jnisvmlight;

import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

/**
 * @author Anja Pilz
 */
public class FeatureVectorTest {

    private static final double ERROR = 0.00001;
    private FeatureVector featureVector1;
    private FeatureVector featureVector2;

    @Before
    public void setUp() throws Exception {
        featureVector1 = new FeatureVector(new int[]{1, 3}, new double[]{1, 1});
        featureVector2 = new FeatureVector(2.0, new int[]{1, 2, 3}, new double[]{1, 1, 2});

    }

    @Test
    public void testCosine() {
        assertThat(featureVector1.getCosine(featureVector1), Matchers.closeTo(1.0, ERROR));
        assertThat(featureVector2.getCosine(featureVector2), Matchers.closeTo(1.0, ERROR));
        assertThat(featureVector1.getCosine(featureVector2), is(3. / (featureVector1.getL2Norm() * featureVector2.getL2Norm())));
        assertThat(featureVector2.getCosine(featureVector1), is(3. / (featureVector1.getL2Norm() * featureVector2.getL2Norm())));
    }

    @Test
    public void testFactor() throws Exception {
        assertThat(featureVector1.getFactor(), is(1.0));
        assertThat(featureVector2.getFactor(), is(2.0));
        featureVector1.setFactor(10.0);
        assertThat(featureVector1.getFactor(), is(10.0));
    }

    @Test
    public void testL1Norm() {
        assertThat(featureVector1.getL1Norm(), is(2.0));
        assertThat(featureVector2.getL1Norm(), is(4.0));
    }

    @Test
    public void testL2Norm() {
        assertThat(featureVector1.getL2Norm(), is(Math.sqrt(2.0)));
        assertThat(featureVector2.getL2Norm(), is(Math.sqrt(6.0)));
    }

    @Test
    public void testL1Normalization() {
        featureVector1.normalizeL1();
        assertThat(featureVector1.m_vals[0], is(0.5));
        assertThat(featureVector1.m_vals[1], is(0.5));

        featureVector2.normalizeL1();
        assertThat(featureVector2.m_vals[0], is(0.25));
        assertThat(featureVector2.m_vals[1], is(0.25));
        assertThat(featureVector2.m_vals[2], is(0.5));
    }

    @Test
    public void testL2Normalization() {
        featureVector1.normalizeL2();
        assertThat(featureVector1.m_vals[0], Matchers.closeTo(0.5, ERROR));
        assertThat(featureVector1.m_vals[1], Matchers.closeTo(0.5, ERROR));

        featureVector2.normalizeL2();
        assertThat(featureVector2.m_vals[0], Matchers.closeTo(1./6., ERROR));
        assertThat(featureVector2.m_vals[1], Matchers.closeTo(1./6., ERROR));
        assertThat(featureVector2.m_vals[2], Matchers.closeTo(4./6., ERROR));
    }

    @Test(expected = IllegalArgumentException.class)
    public void dimensionsAndValuesMustHaveTheSameLength() throws Exception {
        new FeatureVector(new int[]{1}, new double[]{1, 2});
    }

    @Test(expected = IllegalArgumentException.class)
    public void dimensionsMustBeGreaterThanZero() throws Exception {
        new FeatureVector(new int[]{0, 1}, new double[]{1, 2});
    }
}
