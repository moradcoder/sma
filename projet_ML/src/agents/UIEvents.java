/*
 *
 * ------------------------------------------------------------
 * Interface de callback entre les agents JADE et le GUI Swing.
 * Le GUI l'implémente ; le CoordinatorAgent l'appelle.
 * ------------------------------------------------------------
 */
public interface UIEvents {

    /** Log d'un message de communication. */
    void log(String message);

    /** Mise à jour du statut d'un agent (WAITING/RUNNING/TRAINING/...). */
    void agentStatus(String agentName, String status);

    /** Une proposition a été reçue (une ligne dans la JTable). */
    void proposalReceived(String modelName,
                          double accuracy,
                          double precision,
                          double recall,
                          double f1,
                          double time);

    /** Le meilleur modèle a été choisi. */
    void bestModel(String modelName,
                   double accuracy,
                   double precision,
                   double recall,
                   double f1,
                   double time,
                   double score);

    /** Un modèle a été rejeté (pour colorer la JTable). */
    void modelRejected(String modelName);

    /** Simulation terminée (pour réactiver les boutons). */
    void simulationFinished();
}