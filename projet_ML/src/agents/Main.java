/*
 * Main.java
 * ------------------------------------------------------------
 * Point d'entrée.
 *  - Affiche le GUI Swing.
 *  - Quand l'utilisateur clique sur START SIMULATION,
 *    lance JADE + agents + Coordinator lié au GUI.
 * ------------------------------------------------------------
 */

import jade.core.Runtime;
import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.wrapper.AgentContainer;
import jade.wrapper.AgentController;

import javax.swing.SwingUtilities;

public class Main {

    private static AgentContainer container;

    public static void main(String[] args) {

        System.out.println("========================================");
        System.out.println("MULTI-AGENT ML MODEL SELECTION");
        System.out.println("========================================\n");

        // 1. Création du GUI (EDT)
        final GUI[] guiRef = new GUI[1];
        SwingUtilities.invokeLater(() -> {
            GUI gui = new GUI();
            guiRef[0] = gui;

            // Brancher le bouton START
            gui.setOnStartSimulation(() -> startSimulation(gui));

            gui.setVisible(true);
        });
    }

    // ------------------------------------------------------------
    // Démarrage de JADE + agents (dans un thread séparé pour ne
    // pas bloquer l'EDT).
    // ------------------------------------------------------------
    private static void startSimulation(GUI gui) {
        new Thread(() -> {
            try {
                gui.log("[MAIN] Starting JADE platform...");

                Runtime runtime = Runtime.instance();

                Profile profile = new ProfileImpl();
                profile.setParameter(Profile.GUI, "false"); // on cache la GUI JADE
                profile.setParameter(Profile.MAIN_HOST, "localhost");
                profile.setParameter(Profile.MAIN_PORT, "1099");

                container = runtime.createMainContainer(profile);

                // -------- Coordinator (lié au GUI) --------
                Object[] coordArgs = new Object[]{ gui };
                AgentController coordinator = container.createNewAgent(
                        "coordinator", "CoordinatorAgent", coordArgs);

                AgentController logistic = container.createNewAgent(
                        "logistic", "LogisticRegressionAgent", null);
                AgentController decisionTree = container.createNewAgent(
                        "decisionTree", "DecisionTreeAgent", null);
                AgentController neuralNetwork = container.createNewAgent(
                        "neuralNetwork", "NeuralNetworkAgent", null);

                coordinator.start();
                logistic.start();
                decisionTree.start();
                neuralNetwork.start();

                gui.log("[MAIN] 4 agents started. Simulation running...");

            } catch (Exception e) {
                gui.log("[MAIN] ERROR : " + e.getMessage());
                e.printStackTrace();
            }
        }, "JADE-Starter").start();
    }
}