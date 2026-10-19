/*
 * DecisionTreeAgent.java
 * ------------------------------------------------------------
 * Entraîne un arbre CART simplifié et envoie PROPOSE.
 * ------------------------------------------------------------
 */

import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

public class DecisionTreeAgent extends Agent {

    private Dataset dataset;

    private static class Node {
        int featureIndex = -1;
        double threshold;
        Node left, right;
        int label = -1;
    }

    @Override
    protected void setup() {

        System.out.println("[DecisionTreeAgent] Agent started.");
        dataset = new Dataset();
        dataset.load();

        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {

                MessageTemplate template = MessageTemplate.MatchPerformative(ACLMessage.CFP);
                ACLMessage msg = receive(template);

                if (msg != null) {
                    System.out.println("\n[DecisionTreeAgent] CFP received.");
                    System.out.println("[DecisionTreeAgent] Training...");

                    inform(msg.getSender(), "TRAINING");

                    try {
                        long start = System.currentTimeMillis();
                        Node root = buildTree(dataset.getX_train(), dataset.getY_train(), 0);

                        int[] yPred = new int[dataset.getX_test().length];
                        for (int i = 0; i < yPred.length; i++)
                            yPred[i] = predict(root, dataset.getX_test()[i]);

                        int[] yTrue = dataset.getY_test();
                        int nC = Dataset.getNClasses();
                        double acc = Evaluation.accuracy(yTrue, yPred);
                        double prec = Evaluation.precision(yTrue, yPred, nC);
                        double rec = Evaluation.recall(yTrue, yPred, nC);
                        double f1 = Evaluation.f1(yTrue, yPred, nC);

                        long end = System.currentTimeMillis();
                        double time = (end - start) / 1000.0;

                        String content = "Model=Decision Tree"
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

                        System.out.println("[DecisionTreeAgent] PROPOSE sent.");
                        System.out.println("   Accuracy=" + String.format("%.4f", acc)
                                + " | F1=" + String.format("%.4f", f1)
                                + " | Time=" + String.format("%.2f", time) + "s");

                    } catch (Exception e) {
                        ACLMessage refuse = new ACLMessage(ACLMessage.REFUSE);
                        refuse.addReceiver(msg.getSender());
                        refuse.setContent("Training failed: " + e.getMessage());
                        send(refuse);
                        System.err.println("[DecisionTreeAgent] REFUSE : " + e.getMessage());
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
        info.setContent("Agent=decisionTree|State=" + state);
        send(info);
    }

    // ---------------- Build / Predict (inchangé) ----------------
    private Node buildTree(double[][] X, int[] y, int depth) {
        Node node = new Node();
        if (isPure(y) || depth >= 5) {
            node.label = majorityClass(y);
            return node;
        }
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGini = Double.MAX_VALUE;
        int d = X[0].length;

        for (int j = 0; j < d; j++) {
            for (int i = 0; i < X.length; i++) {
                double threshold = X[i][j];
                double gini = giniSplit(X, y, j, threshold);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = j;
                    bestThreshold = threshold;
                }
            }
        }
        if (bestFeature == -1) { node.label = majorityClass(y); return node; }

        int nLeft = 0;
        for (int i = 0; i < X.length; i++) if (X[i][bestFeature] <= bestThreshold) nLeft++;
        int nRight = X.length - nLeft;
        if (nLeft == 0 || nRight == 0) { node.label = majorityClass(y); return node; }

        double[][] Xl = new double[nLeft][d];
        double[][] Xr = new double[nRight][d];
        int[] yl = new int[nLeft];
        int[] yr = new int[nRight];
        int il = 0, ir = 0;
        for (int i = 0; i < X.length; i++) {
            if (X[i][bestFeature] <= bestThreshold) { Xl[il] = X[i]; yl[il] = y[i]; il++; }
            else { Xr[ir] = X[i]; yr[ir] = y[i]; ir++; }
        }
        node.featureIndex = bestFeature;
        node.threshold = bestThreshold;
        node.left = buildTree(Xl, yl, depth + 1);
        node.right = buildTree(Xr, yr, depth + 1);
        return node;
    }

    private boolean isPure(int[] y) {
        for (int i = 1; i < y.length; i++) if (y[i] != y[0]) return false;
        return true;
    }

    private int majorityClass(int[] y) {
        int[] counts = new int[Dataset.getNClasses()];
        for (int v : y) counts[v]++;
        int best = 0;
        for (int c = 1; c < counts.length; c++) if (counts[c] > counts[best]) best = c;
        return best;
    }

    private double giniSplit(double[][] X, int[] y, int feature, double threshold) {
        int[] lc = new int[Dataset.getNClasses()];
        int[] rc = new int[Dataset.getNClasses()];
        int nL = 0, nR = 0;
        for (int i = 0; i < X.length; i++) {
            if (X[i][feature] <= threshold) { lc[y[i]]++; nL++; }
            else { rc[y[i]]++; nR++; }
        }
        double gL = gini(lc, nL);
        double gR = gini(rc, nR);
        return (nL * gL + nR * gR) / (nL + nR);
    }

    private double gini(int[] counts, int total) {
        if (total == 0) return 0;
        double sum = 0;
        for (int c : counts) {
            double p = (double) c / total;
            sum += p * p;
        }
        return 1 - sum;
    }

    private int predict(Node node, double[] x) {
        if (node.label != -1) return node.label;
        if (x[node.featureIndex] <= node.threshold) return predict(node.left, x);
        return predict(node.right, x);
    }
}