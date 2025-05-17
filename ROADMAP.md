# oblix Roadmap (2025)

**Last updated: May 17, 2025**

oblix is a browser-based neural network playground written entirely in plain JavaScript. The app is intentionally self-contained so it can stay lightweight and remain hosted via GitHub Pages. Below is a high-level roadmap outlining planned improvements for the rest of 2025 and into 2026.

## Current Highlights

*   Interactive UI for constructing and training networks directly in the browser.
*   Layer support: Dense, Layer Normalization, Self-Attention, Dropout and Softmax.
*   Architecture templates for MLPs, Autoencoders and Transformer-style blocks.
*   Training options with SGD, Adam, RMSprop and AdamW plus learning rate schedules.
*   Built-in data generation, positional encoding and save/load of models.
*   Real-time visualizations of loss curves and network graphs.

## Near Term Goals (Q2–Q3 2025)

*   **Host on GitHub Pages** – keep deployment simple; update docs so anyone can clone and run `index.html`.
*   **Enhanced documentation** – expand the README with step-by-step tutorials and link to example data sets.
*   **CSV import/export** – drag and drop or select a file to load training data; allow exporting predictions as CSV.
*   **Local storage** for models and training sessions so experiments persist across page reloads.
*   **More visualization options** such as gradient norms or per-layer activation histograms.

## Mid Term Goals (Q4 2025–Q1 2026)

*   **Additional layer types** including simple convolutional and recurrent layers written in pure JS.
*   **Performance tweaks** – explore Web Workers for background computation and typed arrays to speed up math.
*   **Dataset library** with built-in demos showcasing MNIST-like data or synthetic sequence data.
*   **Improved UX**: progress bars, better model summary, and keyboard shortcuts for common actions.

## Long Term Ideas (2026 and beyond)

*   **WebGPU/WebGL acceleration** when available while keeping a graceful fallback to CPU.
*   **Plugin system** so the community can add custom layers, optimizers or visualizations.
*   **Collaborative sharing** of saved models and configurations directly from the browser.
*   **Educational content** including guided tutorials and lesson plans.

