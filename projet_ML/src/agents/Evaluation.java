/*
 * Evaluation.java
 * ------------------------------------------------------------
 * Classe utilitaire (PAS un agent).
 * Calcule Accuracy, Precision, Recall, F1 (macro-average) pour
 * un problème multiclasses.
 * ------------------------------------------------------------
 */

public class Evaluation {

    // ------------------------------------------------------------
    // Accuracy = (nombre de bonnes prédictions) / (total)
    // ------------------------------------------------------------
    public static double accuracy(int[] yTrue, int[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) correct++;
        }
        return (double) correct / yTrue.length;
    }

    // ------------------------------------------------------------
    // Precision macro-average
    //   precision_c = TP_c / (TP_c + FP_c)
    //   precision = moyenne des precision_c
    // ------------------------------------------------------------
    public static double precision(int[] yTrue, int[] yPred, int nClasses) {
        double sum = 0;
        for (int c = 0; c < nClasses; c++) {
            int tp = 0, fp = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yPred[i] == c && yTrue[i] == c) tp++;
                if (yPred[i] == c && yTrue[i] != c) fp++;
            }
            double p = (tp + fp == 0) ? 0 : (double) tp / (tp + fp);
            sum += p;
        }
        return sum / nClasses;
    }

    // ------------------------------------------------------------
    // Recall macro-average
    //   recall_c = TP_c / (TP_c + FN_c)
    //   recall = moyenne des recall_c
    // ------------------------------------------------------------
    public static double recall(int[] yTrue, int[] yPred, int nClasses) {
        double sum = 0;
        for (int c = 0; c < nClasses; c++) {
            int tp = 0, fn = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yPred[i] == c && yTrue[i] == c) tp++;
                if (yPred[i] != c && yTrue[i] == c) fn++;
            }
            double r = (tp + fn == 0) ? 0 : (double) tp / (tp + fn);
            sum += r;
        }
        return sum / nClasses;
    }

    // ------------------------------------------------------------
    // F1-score macro-average
    //   F1 = 2 * (precision * recall) / (precision + recall)
    // ------------------------------------------------------------
    public static double f1(int[] yTrue, int[] yPred, int nClasses) {
        double p = precision(yTrue, yPred, nClasses);
        double r = recall(yTrue, yPred, nClasses);
        if (p + r == 0) return 0;
        return 2 * (p * r) / (p + r);
    }
}