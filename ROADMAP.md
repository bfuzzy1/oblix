# Oblix 2025 Roadmap: The "Awesome" Educational Update

This roadmap outlines the planned evolution of Oblix for the remainder of 2025 (April - December), focusing on delivering an **"awesome" user experience** centered around unique **interactive and educational insights**. Our goal is to solidify Oblix as a go-to tool for visually understanding core machine learning concepts.

## Guiding Principles

*   **Self-Contained:** Remain within pure JS.
*   **Educational:** Prioritize features that aid understanding and provide "Aha!" moments.
*   **Interactive:** Focus on visualization and user control for exploration.
*   **Maintainable:** Ensure code structure supports ongoing development by a solo developer (with AI assistance).

## 2025 Roadmap: Focus on Deep Understanding

The focus for the rest of 2025 is on quality over quantity, delivering features that provide unique learning opportunities.

**Phase 1: Foundation & Core QoL (Target: April - June 2025)**

*   [ ] **[Code Quality] Major Refactoring:** Improve UI/Logic separation and drawing logic for better maintainability and to facilitate future interactive features.
*   [ ] **[Core UX] Pause/Resume Training:** Allow users to pause training, inspect the network state, and resume. Essential for deeper exploration.

**Phase 2: Primary Awesome Feature (Target: July - September 2025)**

*   [ ] **[Interactive Viz] Visual Backpropagation Step-Through:** Implement the ability to visually step through the backpropagation calculation for a single data point, clearly showing gradient flow and weight updates.

**Phase 3: Supporting Feature & Polish (Target: October - December 2025)**

*   [ ] **[Educational Feature] Guided Tutorials Framework:** Implement a system for creating in-tool tutorials.
*   [ ] **[Educational Content] Initial Tutorials:** Create 1-2 tutorials demonstrating core concepts using the new Pause/Resume and Visual Backprop features.
*   [ ] **[Polish] Buffer, Testing & Documentation:** Ensure features are robust, well-tested, and documented.

**Potential Stretch Goal (If time permits in Q4 2025)**

*   [ ] **[Interactive Viz] Side-by-Side Visual Comparison:** Allow running and visualizing two network configurations simultaneously to compare their behavior (e.g., different optimizers, learning rates).

**Future Considerations / Backlog (Post-2025 or Opportunistic)**

*(Ideas moved here from the prioritized list or previous backlog)*
*   Visual Pitfall Illustration Mode: Modes/datasets to visually demo common training problems (vanishing/exploding gradients, overfitting).
*   'What If?' Data Point Analysis: Interactively select data point to see its path, loss contribution, gradients; maybe simulate changes.
*   Implement + Visualize Regularization: Add L1/L2/Dropout and visualize their effect on weights/activations.
*   Shareable "Oblix Snapshots": Export full state (network, weights, data, settings) to a single, interactive HTML file.
*   Weight/Gradient Histograms: Visualize weight/gradient distributions across layers.
*   Plot Validation Metrics on Loss Graph: Add accuracy, R², etc. alongside loss curves on the training graph.
*   Enhanced Tooltips & External Links: Add explanatory tooltips and links to external learning resources in the UI.
*   More Optimizers/Activation Functions: Expand options beyond current defaults, potentially with specific visualizations.
*   Data Preprocessing Options: Add normalization/standardization with potential visualization of effect.
*   Advanced LR Scheduling + Visualization: Implement cosine annealing, cyclical LR, etc., and visualize the schedule.
*   *... and other ideas from brainstorming.*
