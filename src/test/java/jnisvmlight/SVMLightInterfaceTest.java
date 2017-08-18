package jnisvmlight;

import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.nio.file.Paths;
import java.text.ParseException;

public class SVMLightInterfaceTest {

    private SVMLightInterface svmLightInterface;
    private static LabeledFeatureVector[] trainingData;
    private TrainingParameters parameters;
    private static LabeledFeatureVector[] testData;

    @BeforeClass
    public static void importData() throws ParseException {
        // Read training data from file (supplied by T. Joachims)
        trainingData = SVMLightInterface.fromPath(Paths.get("data/example3/train.dat"), 0);
        testData = SVMLightInterface.fromPath(Paths.get("data/example3/test.dat"), 0);
    }

    @Before
    public void setUp() throws Exception {
        // The svmLightInterface with the native communication to the SVM-light shared libraries
        svmLightInterface = new SVMLightInterface();
        // Sort all feature vectors in ascending order of feature dimensions before training the model
        SVMLightInterface.SORT_INPUT_VECTORS = true;
        // Initialize a new TrainingParameters object with the default SVM-light values
        parameters = new TrainingParameters();
        parameters.getLearningParameters().type = LearnParam.RANKING;
    }

    @Test
    public void testTrainModelWithDefaultParameters() throws Exception {
        SVMLightModel model = svmLightInterface.trainModel(trainingData);
        testModel(model);
    }

    @Test
    public void testTrainModelWithStringParameters() throws Exception {
        SVMLightModel model = svmLightInterface.trainModel(trainingData, new String[]{"-r", "1.0", "-z", "p"});
        testModel(model);

    }

    @Test
    public void testTrainModel() throws Exception {
        SVMLightModel model = svmLightInterface.trainModel(trainingData, parameters);
        testModel(model);
    }

    private void testModel(SVMLightModel model) {
        for (LabeledFeatureVector vector : testData) {
            double classifyJni = model.classify(vector);
            double classifyNative = svmLightInterface.classifyNative(vector);
            Assert.assertThat(classifyJni, Matchers.closeTo(classifyNative, 0.0000001));
        }
    }
}