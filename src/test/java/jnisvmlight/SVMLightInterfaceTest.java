package jnisvmlight;

import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;

import static org.junit.Assert.assertThat;

public class SVMLightInterfaceTest {

    private SVMLightInterface svmLightInterface;
    private static LabeledFeatureVector[] trainingData;
    private static LabeledFeatureVector[] testData;


    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

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
    public void testTrainRankingModel() throws Exception {
        // Initialize a new TrainingParameters object
        TrainingParameters parameters = new TrainingParameters();
        parameters.getLearningParameters().type = LearnParam.RANKING;
        SVMLightModel model = svmLightInterface.trainModel(trainingData, parameters);
        testModel(model);
    }

    private void testModel(SVMLightModel model) throws IOException, ParseException {
        for (LabeledFeatureVector vector : trainingData) {
            double classifyJni = model.classify(vector);
            double classifyNative = svmLightInterface.classifyNative(vector);
            assertThat(classifyJni, Matchers.closeTo(classifyNative, 0.00001));
        }
        for (LabeledFeatureVector vector : testData) {
            double classifyJni = model.classify(vector);
            double classifyNative = svmLightInterface.classifyNative(vector);
            assertThat(classifyJni, Matchers.closeTo(classifyNative, 0.00001));
        }
        Path path = folder.newFile().toPath();
        model.writeModelToFile(path.toString());
        SVMLightModel loadedModel = SVMLightModel.fromPath(path);
        for (LabeledFeatureVector vector : testData) {
            double classifyJni = loadedModel.classify(vector);
            double classifyNative = svmLightInterface.classifyNative(vector);
            assertThat(classifyJni, Matchers.closeTo(classifyNative, 0.01));
        }
    }
}