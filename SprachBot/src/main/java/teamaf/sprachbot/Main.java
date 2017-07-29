package teamaf.sprachbot;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;



public class Main
{

    public static final int numchars = 10;
    public static final int layerSize = 300;
    public static final int genders = 3;

    public static void main(String[] args) throws FileNotFoundException
    {
        
        WortIterator iterator = new WortIterator(readData(new File("C:\\Users\\Florian\\Desktop\\nomenliste.txt")));
        DataSet allData = iterator.next(20436);
        //System.out.println(iterator.numExamples());
        //allData.shuffle();
        //SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = iterator.next(20436); //testAndTrain.getTrain();
        DataSet testData = allData; //testAndTrain.getTest();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(10)
                .learningRate(0.1)
                .seed(System.currentTimeMillis())
                .regularization(true)
                .weightInit(WeightInit.RELU)
                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(iterator.inputColumns()).nOut(layerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize)
                        .activation(Activation.TANH).build())
                .layer(3, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize)
                        .activation(Activation.TANH).build())
                .layer(4, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .activation(Activation.TANH)
                        .nIn(layerSize).nOut(genders).build())
                .pretrain(false).backpropType(BackpropType.Standard).backprop(true)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray output;
        ArrayList<String> names = new ArrayList<>(3);
        names.add("m");
        names.add("f");
        names.add("n");
        
        
        Evaluation eval = null;
        for(int i = 0; i < 1000; i++)
        {
            eval = new Evaluation(names);
            System.out.println(i);
            network.fit(trainingData);
            //output = stringToArray("Computer", iterator.charMap());
            //output = network.output(output);
            //System.out.println(output);
            output = network.output(testData.getFeatureMatrix());
            eval.eval(testData.getLabels(), output);
            System.out.println(eval.confusionToString());
            
            allData = iterator.next(20436);
            if(i %15 == 14)
                allData = iterator.next(20436);
            //allData.shuffle();
            //testAndTrain = allData.splitTestAndTrain(0.8);
            trainingData = allData; //testAndTrain.getTrain();
            //ArrayList<DataSet> testing = new ArrayList<>(2);
            //testing.add(testData);
            //testing.add(testAndTrain.getTest());
            //testData = DataSet.merge(testing);
        }
        System.out.println(eval.stats());
        
    }
    
    private static String[] readData(File f) throws FileNotFoundException
    {
        Scanner s = new Scanner(f, "utf-8");
        ArrayList<String> inputs = new ArrayList<>(5000);
        while(s.hasNextLine())
        {
            String str = s.nextLine();
            inputs.add(str.substring(numchars+1) + str.substring(0, numchars));
        }
        String[] output = new String[inputs.size()];
        inputs.toArray(output);
        return output;
    }
    
    private static INDArray stringToArray(String s, HashMap<Character, Integer> map)
    {
        StringBuilder sB = new StringBuilder(s.toUpperCase());
        sB.reverse();
        if(sB.length() > Main.numchars)
            sB.delete(Main.numchars, sB.length());
        else if(sB.length() < Main.numchars)
            while(sB.length() < 10)
                sB.append(' ');
        
        s = sB.toString();
        System.out.println(s);
        
        INDArray inputs = Nd4j.zeros(new int[]
        {
            1, map.size() * Main.numchars
        }, 'f');
        for (int i = 0; i < Main.numchars; i++)
        {
            inputs.putScalar(new int[]
            {
                0, map.get(s.charAt(i)) + map.size() * i
            }, 1.0);
        }
        
        return inputs;
    }
}
