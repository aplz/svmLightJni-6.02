/**
 *
 */
package jnisvmlight;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

/**
 * @author Anja Pilz
 */
public class FeatureVectorTest {

    private FeatureVector featureVector1;
    private FeatureVector featureVector2;

    @Before
    public void setUp() throws Exception {
        featureVector1 = new FeatureVector(new int[]{1, 3}, new double[]{1, 1});
        featureVector2 = new FeatureVector(2.0, new int[]{1, 2, 3}, new double[]{1, 1, 2});

    }

    @Test
    public void testCosine() {
        assertTrue(Math.abs(featureVector1.getCosine(featureVector1) - 1.0) < 0.00001);
        assertTrue(Math.abs(featureVector2.getCosine(featureVector2) - 1.0) < 0.00001);
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
    public void testScalarProduct() {
        //   assertThat(featureVector1.getSprod(featureVector2), is(3.0));
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
