/*
 * NeuralNetworkAgent.java
 * ------------------------------------------------------------
 * Entraîne un MLP 4-8-3 (ReLU + softmax) et envoie PROPOSE.
 * ------------------------------------------------------------
 */

import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

import java.util.Random;

public class NeuralNetworkAgent extends Agent {

    private Dataset dataset;

    @Override
    protected void setup() {

        System.out.println("[NeuralNetworkAgent] Agent started.");
        dataset = new Dataset();
        dataset.load();

        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {

                MessageTemplate template = MessageTemplate.MatchPerformative(ACLMessage.CFP);
                ACLMessage msg = receive(template);

                if (msg != null) {
                    System.out.println("\n[NeuralNetworkAgent] CFP received.");
                    System.out.println("[NeuralNetworkAgent] Training...");

                    inform(msg.getSender(), "TRAINING");

                    try {
                        long start = System.currentTimeMillis();
                        double[][][] weights = train();
                        int[] yPred = predict(weights, dataset.getX_test());

                        int[] yTrue = dataset.getY_test();
                        int nC = Dataset.getNClasses();
                        double acc = Evaluation.accuracy(yTrue, yPred);
                        double prec = Evaluation.precision(yTrue, yPred, nC);
                        double rec = Evaluation.recall(yTrue, yPred, nC);
                        double f1 = Evaluation.f1(yTrue, yPred, nC);

                        long end = System.currentTimeMillis();
                        double time = (end - start) / 1000.0;

                        String content = "Model=Neural Network"
                                + "|Accuracy=" + acc
                                + "|Precision=" + prec
                                + "|Recall=" + rec
                                + "|F1=" + f1
                                + "|Time=" + time;

                        ACLMessage propose = new ACLMessage(ACLMessage.PROPOSE);
                        propose.addReceiver(msg.getSender());
                        propose.setContent(content);
                        propose.setConversationId("ML-CFP");
                        send(propose);

                        System.out.println("[NeuralNetworkAgent] PROPOSE sent.");
                        System.out.println("   Accuracy=" + String.format("%.4f", acc)
                                + " | F1=" + String.format("%.4f", f1)
                                + " | Time=" + String.format("%.2f", time) + "s");

                    } catch (Exception e) {
                        ACLMessage refuse = new ACLMessage(ACLMessage.REFUSE);
                        refuse.addReceiver(msg.getSender());
                        refuse.setContent("Training failed: " + e.getMessage());
                        send(refuse);
                        System.err.println("[NeuralNetworkAgent] REFUSE : " + e.getMessage());
                    }
                } else {
                    block();
                }
            }
        });
    }

    private void inform(jade.core.AID coordinator, String state) {
        ACLMessage info = new ACLMessage(ACLMessage.INFORM);
        info.addReceiver(coordinator);
        info.setOntology("AGENT-STATUS");
        info.setContent("Agent=neuralNetwork|State=" + state);
        send(info);
    }

    // ---------------- Train / Predict (inchangé) ----------------
    private double[][][] train() {
        double[][] X = dataset.getX_train();
        int[] y = dataset.getY_train();
        int n = X.length;
        int d = Dataset.getNFeatures();
        int h = 8;
        int k = Dataset.getNClasses();

        Random rand = new Random(42);
        double[][] W1 = new double[h][d];
        double[] b1 = new double[h];
        double[][] W2 = new double[k][h];
        double[] b2 = new double[k];

        for (int i = 0; i < h; i++)
            for (int j = 0; j < d; j++) W1[i][j] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < h; j++) W2[i][j] = rand.nextGaussian() * 0.1;

        double lr = 0.1;
        int epochs = 500;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                double[] z1 = new double[h];
                double[] a1 = new double[h];
                for (int j = 0; j < h; j++) {
                    for (int t = 0; t < d; t++) z1[j] += W1[j][t] * X[i][t];
                    z1[j] += b1[j];
                    a1[j] = Math.max(0, z1[j]);
                }

                double[] z2 = new double[k];
                for (int c = 0; c < k; c++) {
                    for (int j = 0; j < h; j++) z2[c] += W2[c][j] * a1[j];
                    z2[c] += b2[c];
                }

                double max = z2[0];
                for (int c = 1; c < k; c++) max = Math.max(max, z2[c]);
                double sum = 0;
                double[] p = new double[k];
                for (int c = 0; c < k; c++) { p[c] = Math.exp(z2[c] - max); sum += p[c]; }
                for (int c = 0; c < k; c++) p[c] /= sum;

                double[] dz2 = new double[k];
                for (int c = 0; c < k; c++) dz2[c] = p[c] - (y[i] == c ? 1 : 0);

                double[] da1 = new double[h];
                for (int j = 0; j < h; j++)
                    for (int c = 0; c < k; c++) da1[j] += dz2[c] * W2[c][j];

                double[] dz1 = new double[h];
                for (int j = 0; j < h; j++) dz1[j] = (z1[j] > 0) ? da1[j] : 0;

                for (int c = 0; c < k; c++) {
                    for (int j = 0; j < h; j++) W2[c][j] -= lr * dz2[c] * a1[j];
                    b2[c] -= lr * dz2[c];
                }
                for (int j = 0; j < h; j++) {
                    for (int t = 0; t < d; t++) W1[j][t] -= lr * dz1[j] * X[i][t];
                    b1[j] -= lr * dz1[j];
                }
            }
        }
        return new double[][][]{W1, W2};
    }

    private int[] predict(double[][][] weights, double[][] X) {
        double[][] W1 = weights[0];
        double[][] W2 = weights[1];
        int h = W1.length;
        int k = W2.length;
        int[] yPred = new int[X.length];

        for (int i = 0; i < X.length; i++) {
            double[] a1 = new double[h];
            for (int j = 0; j < h; j++) {
                for (int t = 0; t < X[i].length; t++) a1[j] += W1[j][t] * X[i][t];
                a1[j] = Math.max(0, a1[j]);
            }
            double[] z2 = new double[k];
            for (int c = 0; c < k; c++)
                for (int j = 0; j < h; j++) z2[c] += W2[c][j] * a1[j];
            int best = 0;
            for (int c = 1; c < k; c++) if (z2[c] > z2[best]) best = c;
            yPred[i] = best;
        }
        return yPred;
    }
}