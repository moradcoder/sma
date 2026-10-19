/*
 * 
 * ------------------------------------------------------------
 * Interface Swing professionnelle pour le système multi-agents.
 * Implémente UIEvents et reçoit les callbacks du CoordinatorAgent.
 * ------------------------------------------------------------
 */

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;

public class GUI extends JFrame implements UIEvents {

    // ---------- Couleurs ----------
    private static final Color BG_DARK      = new Color(30, 34, 45);
    private static final Color BG_PANEL     = new Color(40, 45, 60);
    private static final Color BG_HEADER    = new Color(24, 28, 38);
    private static final Color ACCENT       = new Color(0, 200, 180);
    private static final Color ACCENT2      = new Color(255, 180, 0);
    private static final Color TEXT_LIGHT   = new Color(230, 235, 245);
    private static final Color TEXT_DIM     = new Color(160, 170, 185);
    private static final Color GREEN        = new Color(80, 200, 120);
    private static final Color RED          = new Color(230, 90, 90);
    private static final Color ORANGE       = new Color(255, 160, 60);

    // ---------- Composants ----------
    private final Map<String, JLabel> agentStatusLabels = new HashMap<>();
    private DefaultTableModel tableModel;
    private JTable resultTable;

    private JLabel bestModelName;
    private JLabel bestAccuracy;
    private JLabel bestF1;
    private JLabel bestTime;
    private JLabel bestScore;

    private JTextArea logArea;
    private JButton startButton;
    private JButton resetButton;
    private JButton exitButton;

    // Callback vers Main (démarrage de la simulation)
    private Runnable onStartSimulation;

    // ---------- Constructeur ----------
    public GUI() {
        super("Multi-Agent ML Model Selection — JADE");
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setSize(1050, 820);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout());
        getContentPane().setBackground(BG_DARK);

        add(buildHeader(), BorderLayout.NORTH);
        add(buildCenter(), BorderLayout.CENTER);
        add(buildFooter(), BorderLayout.SOUTH);

        // Gestion fermeture
        addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent e) {
                doExit();
            }
        });
    }

    public void setOnStartSimulation(Runnable r) { this.onStartSimulation = r; }

    // ============================================================
    //  HEADER
    // ============================================================
    private JPanel buildHeader() {
        JPanel header = new JPanel();
        header.setLayout(new BoxLayout(header, BoxLayout.Y_AXIS));
        header.setBackground(BG_HEADER);
        header.setBorder(new EmptyBorder(18, 20, 18, 20));

        JLabel title = new JLabel("MULTI-AGENT ML MODEL SELECTION");
        title.setForeground(ACCENT);
        title.setFont(new Font("Segoe UI", Font.BOLD, 22));
        title.setAlignmentX(Component.CENTER_ALIGNMENT);

        JLabel subtitle = new JLabel("JADE Intelligent Model Selection System");
        subtitle.setForeground(TEXT_DIM);
        subtitle.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        subtitle.setAlignmentX(Component.CENTER_ALIGNMENT);

        header.add(title);
        header.add(Box.createVerticalStrut(4));
        header.add(subtitle);
        return header;
    }

    // ============================================================
    //  CENTER : status + table + best model + log
    // ============================================================
    private JPanel buildCenter() {
        JPanel center = new JPanel(new BorderLayout(10, 10));
        center.setBackground(BG_DARK);
        center.setBorder(new EmptyBorder(12, 12, 12, 12));

        // --- Gauche : Agent status ---
        JPanel left = new JPanel(new BorderLayout());
        left.setBackground(BG_DARK);
        left.add(buildAgentStatusPanel(), BorderLayout.NORTH);
        left.add(buildLogPanel(), BorderLayout.CENTER);

        // --- Droite : Table + Best model ---
        JPanel right = new JPanel(new BorderLayout(10, 10));
        right.setBackground(BG_DARK);
        right.add(buildTablePanel(), BorderLayout.CENTER);
        right.add(buildBestModelPanel(), BorderLayout.SOUTH);

        // Split horizontal
        JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, left, right);
        split.setResizeWeight(0.42);
        split.setDividerSize(6);
        split.setBorder(null);
        split.setBackground(BG_DARK);

        center.add(split, BorderLayout.CENTER);
        return center;
    }

    // ---------- Agent Status Panel ----------
    private JPanel buildAgentStatusPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBackground(BG_PANEL);
        panel.setBorder(compoundBorder("AGENT STATUS"));

        String[][] agents = {
                {"coordinator", "Coordinator"},
                {"logistic", "Logistic Regression"},
                {"decisionTree", "Decision Tree"},
                {"neuralNetwork", "Neural Network"}
        };

        for (String[] a : agents) {
            JPanel row = new JPanel(new BorderLayout());
            row.setBackground(BG_PANEL);
            row.setBorder(new EmptyBorder(6, 12, 6, 12));

            JLabel name = new JLabel("●  " + a[1]);
            name.setForeground(TEXT_LIGHT);
            name.setFont(new Font("Segoe UI", Font.PLAIN, 13));

            JLabel status = new JLabel("WAITING");
            status.setForeground(TEXT_DIM);
            status.setFont(new Font("Segoe UI", Font.BOLD, 12));
            status.setHorizontalAlignment(SwingConstants.RIGHT);

            agentStatusLabels.put(a[0], status);

            row.add(name, BorderLayout.WEST);
            row.add(status, BorderLayout.EAST);
            panel.add(row);
        }
        return panel;
    }

    // ---------- Log Panel ----------
    private JPanel buildLogPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(BG_PANEL);
        panel.setBorder(compoundBorder("COMMUNICATION LOG"));

        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setBackground(new Color(22, 26, 34));
        logArea.setForeground(new Color(180, 230, 200));
        logArea.setFont(new Font("Consolas", Font.PLAIN, 12));
        logArea.setMargin(new Insets(8, 10, 8, 10));

        JScrollPane scroll = new JScrollPane(logArea);
        scroll.setBorder(null);
        panel.add(scroll, BorderLayout.CENTER);

        panel.setPreferredSize(new Dimension(0, 260));
        return panel;
    }

    // ---------- Table Panel ----------
    private JPanel buildTablePanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBackground(BG_PANEL);
        panel.setBorder(compoundBorder("MODEL COMPARISON"));

        String[] cols = {"Model", "Accuracy", "F1 Score", "Time (s)", "Global Score", "Status"};
        tableModel = new DefaultTableModel(cols, 0) {
            @Override public boolean isCellEditable(int r, int c) { return false; }
        };

        resultTable = new JTable(tableModel);
        resultTable.setBackground(new Color(22, 26, 34));
        resultTable.setForeground(TEXT_LIGHT);
        resultTable.setGridColor(new Color(60, 68, 85));
        resultTable.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        resultTable.setRowHeight(28);
        resultTable.getTableHeader().setBackground(BG_HEADER);
        resultTable.getTableHeader().setForeground(ACCENT);
        resultTable.getTableHeader().setFont(new Font("Segoe UI", Font.BOLD, 13));
        resultTable.setSelectionBackground(new Color(0, 120, 120));

        // Renderer pour colorer la colonne Status
        resultTable.getColumnModel().getColumn(5).setCellRenderer(new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable t, Object v,
                    boolean sel, boolean foc, int row, int col) {
                Component c = super.getTableCellRendererComponent(t, v, sel, foc, row, col);
                String s = String.valueOf(v);
                if ("ACCEPTED".equals(s)) c.setForeground(GREEN);
                else if ("REJECTED".equals(s)) c.setForeground(RED);
                else if ("PROPOSED".equals(s)) c.setForeground(ORANGE);
                else c.setForeground(TEXT_LIGHT);
                return c;
            }
        });

        JScrollPane scroll = new JScrollPane(resultTable);
        scroll.setBorder(null);
        scroll.getViewport().setBackground(new Color(22, 26, 34));
        panel.add(scroll, BorderLayout.CENTER);

        return panel;
    }

    // ---------- Best Model Panel ----------
    private JPanel buildBestModelPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBackground(BG_PANEL);
        panel.setBorder(compoundBorder("BEST MODEL"));

        GridBagConstraints g = new GridBagConstraints();
        g.insets = new Insets(4, 14, 4, 14);
        g.anchor = GridBagConstraints.WEST;

        bestModelName = new JLabel("—");
        bestModelName.setForeground(ACCENT2);
        bestModelName.setFont(new Font("Segoe UI", Font.BOLD, 20));

        g.gridx = 0; g.gridy = 0; g.gridwidth = 2;
        panel.add(bestModelName, g);
        g.gridwidth = 1;

        bestAccuracy = valueRow(panel, g, 1, "Accuracy");
        bestF1       = valueRow(panel, g, 2, "F1 Score");
        bestTime     = valueRow(panel, g, 3, "Training Time");
        bestScore    = valueRow(panel, g, 4, "Global Score");

        return panel;
    }

    private JLabel valueRow(JPanel p, GridBagConstraints g, int row, String label) {
        JLabel lbl = new JLabel(label + " :");
        lbl.setForeground(TEXT_DIM);
        lbl.setFont(new Font("Segoe UI", Font.PLAIN, 12));

        JLabel val = new JLabel("—");
        val.setForeground(TEXT_LIGHT);
        val.setFont(new Font("Segoe UI", Font.BOLD, 13));

        g.gridx = 0; g.gridy = row;
        p.add(lbl, g);
        g.gridx = 1;
        p.add(val, g);
        return val;
    }

    // ============================================================
    //  FOOTER
    // ============================================================
    private JPanel buildFooter() {
        JPanel footer = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 12));
        footer.setBackground(BG_HEADER);

        startButton = styledButton("START SIMULATION", ACCENT, BG_HEADER);
        resetButton = styledButton("RESET", ACCENT2, BG_HEADER);
        exitButton  = styledButton("EXIT", RED, BG_HEADER);

        startButton.addActionListener(e -> {
            startButton.setEnabled(false);
            resetButton.setEnabled(true);
            resetUI();
            if (onStartSimulation != null) onStartSimulation.run();
        });

        resetButton.addActionListener(e -> {
            resetUI();
            startButton.setEnabled(true);
        });

        exitButton.addActionListener(e -> doExit());

        footer.add(startButton);
        footer.add(resetButton);
        footer.add(exitButton);
        return footer;
    }

    private JButton styledButton(String text, Color fg, Color bg) {
        JButton b = new JButton(text);
        b.setForeground(fg);
        b.setBackground(bg);
        b.setFont(new Font("Segoe UI", Font.BOLD, 13));
        b.setFocusPainted(false);
        b.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(fg, 1),
                new EmptyBorder(8, 22, 8, 22)));
        b.setCursor(new Cursor(Cursor.HAND_CURSOR));
        return b;
    }

    // ============================================================
    //  Utilitaires
    // ============================================================
    private Border compoundBorder(String title) {
        TitledBorder tb = BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(70, 80, 100), 1),
                title);
        tb.setTitleColor(ACCENT);
        tb.setTitleFont(new Font("Segoe UI", Font.BOLD, 12));
        return BorderFactory.createCompoundBorder(tb, new EmptyBorder(6, 6, 6, 6));
    }

    private void resetUI() {
        for (JLabel l : agentStatusLabels.values()) {
            l.setText("WAITING");
            l.setForeground(TEXT_DIM);
        }
        tableModel.setRowCount(0);
        logArea.setText("");
        bestModelName.setText("—");
        bestAccuracy.setText("—");
        bestF1.setText("—");
        bestTime.setText("—");
        bestScore.setText("—");
    }

    private void doExit() {
        int r = JOptionPane.showConfirmDialog(this,
                "Exit the application?", "Confirm",
                JOptionPane.YES_NO_OPTION);
        if (r == JOptionPane.YES_OPTION) {
            System.exit(0);
        }
    }

    // ============================================================
    //  Implémentation UIEvents (appelée depuis threads JADE)
    // ============================================================
    @Override
    public void log(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }

    @Override
    public void agentStatus(String agentName, String status) {
        SwingUtilities.invokeLater(() -> {
            JLabel l = agentStatusLabels.get(agentName);
            if (l == null) return;
            l.setText(status);
            switch (status) {
                case "RUNNING":   l.setForeground(ACCENT); break;
                case "TRAINING":  l.setForeground(ORANGE); break;
                case "PROPOSING": l.setForeground(ACCENT2); break;
                case "ACCEPTED":  l.setForeground(GREEN); break;
                case "REJECTED":  l.setForeground(RED); break;
                case "FINISHED":  l.setForeground(GREEN); break;
                default:          l.setForeground(TEXT_DIM);
            }
        });
    }

    @Override
    public void proposalReceived(String modelName, double accuracy, double precision,
                                 double recall, double f1, double time) {
        SwingUtilities.invokeLater(() -> {
            tableModel.addRow(new Object[]{
                    modelName,
                    String.format("%.4f", accuracy),
                    String.format("%.4f", f1),
                    String.format("%.4f", time),
                    "—",
                    "PROPOSED"
            });
        });
    }

    @Override
    public void bestModel(String modelName, double accuracy, double precision,
                          double recall, double f1, double time, double score) {
        SwingUtilities.invokeLater(() -> {
            bestModelName.setText(modelName);
            bestAccuracy.setText(String.format("%.2f %%", accuracy * 100));
            bestF1.setText(String.format("%.2f %%", f1 * 100));
            bestTime.setText(String.format("%.4f s", time));
            bestScore.setText(String.format("%.4f", score));

            // Mise à jour de la ligne correspondante dans la table
            for (int i = 0; i < tableModel.getRowCount(); i++) {
                if (modelName.equals(tableModel.getValueAt(i, 0))) {
                    tableModel.setValueAt(String.format("%.4f", score), i, 4);
                    tableModel.setValueAt("ACCEPTED", i, 5);
                }
            }
        });
    }

    @Override
    public void modelRejected(String modelName) {
        SwingUtilities.invokeLater(() -> {
            for (int i = 0; i < tableModel.getRowCount(); i++) {
                if (modelName.equals(tableModel.getValueAt(i, 0))) {
                    tableModel.setValueAt("REJECTED", i, 5);
                }
            }
        });
    }

    @Override
    public void simulationFinished() {
        SwingUtilities.invokeLater(() -> {
            startButton.setEnabled(true);
        });
    }
}