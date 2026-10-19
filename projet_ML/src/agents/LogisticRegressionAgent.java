/*
 * LogisticRegressionAgent.java
 * ------------------------------------------------------------
 * Entraîne une régression logistique softmax et envoie PROPOSE.
 * ------------------------------------------------------------
 */

import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

public class LogisticRegressionAgent extends Agent {

    private Dataset dataset;

    @Override
    protected void setup() {

        System.out.println("[LogisticRegressionAgent] Agent started.");

        dataset = new Dataset();
        dataset.load();

        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {

                MessageTemplate template = MessageTemplate.MatchPerformative(ACLMessage.CFP);
                ACLMessage msg = receive(template);

                if (msg != null) {
                    System.out.println("\n[LogisticRegressionAgent] CFP received.");
                    System.out.println("[LogisticRegressionAgent] Training...");

                    // -------- Notification GUI (début) --------
                    inform(msg.getSender(), "TRAINING");

                    try {
                        long start = System.currentTimeMillis();
                        double[][] W = train();
                        int[] yPred = predict(W, dataset.getX_test());

                        int[] yTrue = dataset.getY_test();
                        int nC = Dataset.getNClasses();

                        double acc = Evaluation.accuracy(yTrue, yPred);
                        double prec = Evaluation.precision(yTrue, yPred, nC);
                        double rec = Evaluation.recall(yTrue, yPred, nC);
                        double f1 = Evaluation.f1(yTrue, yPred, nC);

                        long end = System.currentTimeMillis();
                        double time = (end - start) / 1000.0;

                        String content = "Model=Logistic Regression"
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

                        System.out.println("[LogisticRegressionAgent] PROPOSE sent.");
                        System.out.println("   Accuracy=" + String.format("%.4f", acc)
                                + " | F1=" + String.format("%.4f", f1)
                                + " | Time=" + String.format("%.2f", time) + "s");

                    } catch (Exception e) {
                        ACLMessage refuse = new ACLMessage(ACLMessage.REFUSE);
                        refuse.addReceiver(msg.getSender());
                        refuse.setContent("Training failed: " + e.getMessage());
                        send(refuse);
                        System.err.println("[LogisticRegressionAgent] REFUSE : " + e.getMessage());
                    }
                } else {
                    block();
                }
            }
        });
    }

    // Envoi d'un petit INFORM pour signaler l'état TRAINING au Coordinator
    private void inform(jade.core.AID coordinator, String state) {
        ACLMessage info = new ACLMessage(ACLMessage.INFORM);
        info.addReceiver(coordinator);
        info.setOntology("AGENT-STATUS");
        info.setContent("Agent=logistic|State=" + state);
        send(info);
    }

    // ---------------- Training (inchangé) ----------------
    private double[][] train() {
        double[][] X = dataset.getX_train();
        int[] y = dataset.getY_train();
        int n = X.length;
        int d = Dataset.getNFeatures();
        int k = Dataset.getNClasses();

        double[][] W = new double[k][d];
        double lr = 0.1;
        int epochs = 1000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[][] grad = new double[k][d];
            for (int i = 0; i < n; i++) {
                double[] z = new double[k];
                for (int c = 0; c < k; c++)
                    for (int j = 0; j < d; j++) z[c] += W[c][j] * X[i][j];

                double max = z[0];
                for (int c = 1; c < k; c++) max = Math.max(max, z[c]);
                double sum = 0;
                double[] p = new double[k];
                for (int c = 0; c < k; c++) {
                    p[c] = Math.exp(z[c] - max);
                    sum += p[c];
                }
                for (int c = 0; c < k; c++) p[c] /= sum;

                for (int c = 0; c < k; c++) {
                    double error = p[c] - (y[i] == c ? 1 : 0);
                    for (int j = 0; j < d; j++) grad[c][j] += error * X[i][j];
                }
            }
            for (int c = 0; c < k; c++)
                for (int j = 0; j < d; j++)
                    W[c][j] -= lr * (grad[c][j] / n);
        }
        return W;
    }

    private int[] predict(double[][] W, double[][] X) {
        int n = X.length;
        int d = Dataset.getNFeatures();
        int k = Dataset.getNClasses();
        int[] yPred = new int[n];
        for (int i = 0; i < n; i++) {
            double[] z = new double[k];
            for (int c = 0; c < k; c++)
                for (int j = 0; j < d; j++) z[c] += W[c][j] * X[i][j];
            int best = 0;
            for (int c = 1; c < k; c++) if (z[c] > z[best]) best = c;
            yPred[i] = best;
        }
        return yPred;
    }
}