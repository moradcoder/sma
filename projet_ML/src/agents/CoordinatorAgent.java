/*
 * CoordinatorAgent.java
 * ------------------------------------------------------------
 * Rôle : lancer le Contract Net Protocol et notifier le GUI
 *        à chaque étape via UIEvents.
 * ------------------------------------------------------------
 */

import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.core.behaviours.OneShotBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

import java.util.ArrayList;
import java.util.List;

public class CoordinatorAgent extends Agent {

    private AID[] mlAgents;
    private List<Proposal> proposals = new ArrayList<>();
    private static final int N_AGENTS = 3;

    // Référence vers le GUI (reçue dans les arguments)
    private UIEvents ui;

    private static class Proposal {
        AID sender;
        String modelName;
        double accuracy, precision, recall, f1, trainingTime, score;

        @Override
        public String toString() {
            return modelName + " [Acc=" + String.format("%.3f", accuracy)
                    + ", F1=" + String.format("%.3f", f1)
                    + ", Time=" + String.format("%.2f", trainingTime)
                    + "s, Score=" + String.format("%.3f", score) + "]";
        }
    }

    @Override
    protected void setup() {

        // Récupération du GUI depuis les arguments
        Object[] args = getArguments();
        if (args != null && args.length > 0 && args[0] instanceof UIEvents) {
            ui = (UIEvents) args[0];
        }

        if (ui != null) {
            ui.log("[Coordinator] Agent started.");
            ui.agentStatus("coordinator", "RUNNING");
            ui.agentStatus("logistic", "RUNNING");
            ui.agentStatus("decisionTree", "RUNNING");
            ui.agentStatus("neuralNetwork", "RUNNING");
        }

        mlAgents = new AID[]{
                new AID("logistic", AID.ISLOCALNAME),
                new AID("decisionTree", AID.ISLOCALNAME),
                new AID("neuralNetwork", AID.ISLOCALNAME)
        };

        // 1. Envoi du CFP
        addBehaviour(new OneShotBehaviour() {
            @Override
            public void action() {
                if (ui != null) ui.log("[Coordinator] Sending CFP...");
                System.out.println("\n[Coordinator] Sending CFP...");

                ACLMessage cfp = new ACLMessage(ACLMessage.CFP);
                for (AID a : mlAgents) cfp.addReceiver(a);
                cfp.setContent("Please train your model and propose your performance.");
                cfp.setConversationId("ML-CFP");
                send(cfp);
            }
        });

        // 2. Réception des PROPOSE
        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                MessageTemplate template = MessageTemplate.MatchPerformative(ACLMessage.PROPOSE);
                ACLMessage msg = receive(template);

                if (msg != null) {
                    try {
                        Proposal p = parseProposal(msg);
                        proposals.add(p);

                        String senderName = msg.getSender().getLocalName();
                        System.out.println("[Coordinator] PROPOSE received from " + senderName);
                        System.out.println("   " + p);

                        if (ui != null) {
                            ui.log("[Coordinator] PROPOSE received from " + senderName);
                            ui.agentStatus(senderName, "PROPOSING");
                            ui.proposalReceived(p.modelName, p.accuracy, p.precision,
                                    p.recall, p.f1, p.trainingTime);
                        }

                        if (proposals.size() == N_AGENTS) {
                            decide();
                        }
                    } catch (Exception e) {
                        System.err.println("[Coordinator] Erreur parsing : " + e.getMessage());
                    }
                } else {
                    block();
                }
            }
        });
    }

    private Proposal parseProposal(ACLMessage msg) {
        Proposal p = new Proposal();
        p.sender = msg.getSender();
        for (String part : msg.getContent().split("\\|")) {
            String[] kv = part.split("=");
            switch (kv[0]) {
                case "Model":     p.modelName = kv[1]; break;
                case "Accuracy":  p.accuracy = Double.parseDouble(kv[1]); break;
                case "Precision": p.precision = Double.parseDouble(kv[1]); break;
                case "Recall":    p.recall = Double.parseDouble(kv[1]); break;
                case "F1":        p.f1 = Double.parseDouble(kv[1]); break;
                case "Time":      p.trainingTime = Double.parseDouble(kv[1]); break;
            }
        }
        return p;
    }

    private void decide() {

        if (ui != null) ui.log("[Coordinator] Comparing proposals...");
        System.out.println("\n[Coordinator] Comparing proposals...");

        double maxTime = 0;
        for (Proposal p : proposals) maxTime = Math.max(maxTime, p.trainingTime);
        if (maxTime == 0) maxTime = 1;

        for (Proposal p : proposals) {
            double normTime = p.trainingTime / maxTime;
            p.score = 0.5 * p.accuracy + 0.4 * p.f1 - 0.1 * normTime;
        }

        Proposal best = proposals.get(0);
        for (Proposal p : proposals) if (p.score > best.score) best = p;

        System.out.println("\n[Coordinator] BEST MODEL = " + best.modelName);
        if (ui != null) ui.log("[Coordinator] BEST MODEL = " + best.modelName);

        // ACCEPT_PROPOSAL au gagnant
        ACLMessage accept = new ACLMessage(ACLMessage.ACCEPT_PROPOSAL);
        accept.addReceiver(best.sender);
        accept.setContent("Accepted: " + best.modelName);
        accept.setConversationId("ML-RESULT");
        send(accept);
        System.out.println("[Coordinator] ACCEPT_PROPOSAL sent to " + best.sender.getLocalName());
        if (ui != null) ui.log("[Coordinator] ACCEPT_PROPOSAL sent to " + best.sender.getLocalName());

        // REJECT_PROPOSAL aux autres
        for (Proposal p : proposals) {
            if (p != best) {
                ACLMessage reject = new ACLMessage(ACLMessage.REJECT_PROPOSAL);
                reject.addReceiver(p.sender);
                reject.setContent("Rejected: " + p.modelName);
                reject.setConversationId("ML-RESULT");
                send(reject);
                System.out.println("[Coordinator] REJECT_PROPOSAL sent to " + p.sender.getLocalName());
                if (ui != null) {
                    ui.log("[Coordinator] REJECT_PROPOSAL sent to " + p.sender.getLocalName());
                    ui.agentStatus(p.sender.getLocalName(), "REJECTED");
                    ui.modelRejected(p.modelName);
                }
            }
        }

        // INFORM final
        String result = "BestModel=" + best.modelName
                + "|Accuracy=" + best.accuracy
                + "|F1=" + best.f1
                + "|Time=" + best.trainingTime
                + "|Score=" + best.score;

        ACLMessage inform = new ACLMessage(ACLMessage.INFORM);
        for (AID a : mlAgents) inform.addReceiver(a);
        inform.setContent(result);
        inform.setConversationId("ML-FINAL");
        send(inform);

        // Mise à jour du GUI
        if (ui != null) {
            ui.agentStatus(best.sender.getLocalName(), "ACCEPTED");
            ui.bestModel(best.modelName, best.accuracy, best.precision,
                    best.recall, best.f1, best.trainingTime, best.score);

            System.out.println("\n========================================");
            System.out.println("FINAL RESULT");
            System.out.println("========================================");
            System.out.println("Best Model   : " + best.modelName);
            System.out.println("Accuracy     : " + String.format("%.4f", best.accuracy));
            System.out.println("F1           : " + String.format("%.4f", best.f1));
            System.out.println("Training Time: " + String.format("%.4f", best.trainingTime) + " s");
            System.out.println("Global Score : " + String.format("%.4f", best.score));
            System.out.println("========================================\n");

            ui.log("[Coordinator] FINAL RESULT = " + best.modelName);
            ui.simulationFinished();
        }
    }
}