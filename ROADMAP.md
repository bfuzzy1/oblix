# Oblix 2025 Roadmap

This roadmap outlines planned features and improvements for Oblix, focusing on enhancing the core engine, improving user experience, and maintaining code quality within the single-file constraint.

## Guiding Principles

*   **Self-Contained:** Remain within `index.html` with pure JS.
*   **Educational:** Prioritize features that aid understanding.
*   **Interactive:** Focus on visualization and user control.
*   **Maintainable:** Improve code structure for future development.

## Q2/Q3 2025: Foundational Improvements

*   **[Core NN] Configurable Weight Initialization:**
    *   Allow users to select between current (Glorot-like), He, or potentially others.
*   **[UX] Configurable Save Filenames:**
    *   Prompt user for a filename when saving the model.
*   **[UX] Sample Datasets:**
    *   Add built-in options to load classic datasets like XOR.
*   **[UX] Improved Input Validation:**
    *   Enhance pre-training checks for configuration errors (e.g., layer size mismatches, invalid parameters). Provide clearer error messages.
*   **[Code Quality] UI/Logic Separation Refactor:**
    *   Begin refactoring JavaScript to better separate DOM manipulation/event handling from the core `oblix` class logic.

## Future Considerations (Backlog / Potential Q4+)

*   Plot Validation Metrics on Loss Graph
*   Weight/Gradient Histograms
*   Enhanced Tooltips & External Links
*   Pause/Resume Training
*   More Optimizers/Activation Functions
*   Data Preprocessing Options
*   Advanced LR Scheduling
*   Refactor Drawing Logic
*   *See brainstorm list for more ideas...*