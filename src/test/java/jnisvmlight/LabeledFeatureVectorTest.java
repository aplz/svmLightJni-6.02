package jnisvmlight;

import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

/**
 * @author Anja Pilz
 */
public class LabeledFeatureVectorTest {


    private LabeledFeatureVector vector;

    @Before
    public void setUp() throws Exception {
        int[] dimensions = new int[]{1, 2};
        double[] values = new double[]{1., 2.};
        vector = new LabeledFeatureVector(1.0, dimensions, values);

    }

    @Test
    public void testGetLabel() throws Exception {
        assertThat(vector.getLabel(), is(1.0));
    }

    @Test
    public void testQueryId() throws Exception {
        vector.setQueryId(2);
        assertThat(vector.getQueryId(), is(2));
    }
}