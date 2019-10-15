/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml_linearregression;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author Student
 */
public class ML_LinearRegression {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        process("iceCreamSale_Train.arff", "iceCreamSale_Test.arff", "iceCreamSale_Predict.arff");

    }

    public static Instances getDataSet(String filename) {

        try {
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(filename));
            Instances dataSet = loader.getDataSet();
            dataSet.setClassIndex(1);
            return dataSet;

        } catch (IOException ex) {
            Logger.getLogger(ML_LinearRegression.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    public static void process(String trainFilename, String testFilename, String predictFilename) {

        try {
            Instances trainDataSet = getDataSet(trainFilename);
            Instances testDataSet = getDataSet(testFilename);

            Classifier classifier = new LinearRegression();
            classifier.buildClassifier(trainDataSet);
            Evaluation evaluation1 = new Evaluation(trainDataSet);
            evaluation1.evaluateModel(classifier, testDataSet);

            System.out.println("Linear Regression");
            System.out.println(evaluation1.toSummaryString());

            System.out.println("Expression for input data");
            System.out.println(classifier);
            
            Instance predicDataSet = getDataSet(predictFilename).lastInstance();
            double value = classifier.classifyInstance(predicDataSet);
            System.out.println("Predict sale of tempurater 15.2 is ");
            System.out.println(value);
            

        } catch (Exception ex) {
            Logger.getLogger(ML_LinearRegression.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

}
