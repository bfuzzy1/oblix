# oblix

**oblix** is a self-contained, browser-based neural network playground written entirely in pure JavaScript. It provides an interactive environment to build, train, visualize, and experiment with various neural network architectures and training parameters. The main HTML page loads JavaScript modules from the `src` directory.

## Features

*   **Interactive UI:** Build networks and configure training through a graphical interface.
*   **Multiple Layer Types:** Supports Dense (fully connected), Layer Normalization, Self-Attention, Dropout, and Softmax layers.
*   **Architecture Templates:** Quickly load predefined structures like Simple MLP, Basic Autoencoder, Transformer Encoder block, Residual Attention block, and additional presets including MLP with Dropout, Deep Residual MLP, Transformer Stack, Autoencoder with Dropout, and a Softmax Classifier.
*   **Configurable Training:**
    *   **Optimizers:** SGD, Adam, RMSprop, AdamW.
    *   **Loss Functions:** Mean Squared Error (MSE), Cross-Entropy.
    *   **Learning Rate:** Set initial rate and optional schedules (Step Decay, Exponential Decay).
    *   **Regularization:** L2 Weight Decay.
    *   **Gradient Clipping:** Prevent exploding gradients.
    *   **Batch Size:** Control the number of samples per update.
*   **Positional Encoding:** Option to add positional information to inputs, useful for sequence-like data.
*   **Built-in Data Generation:** Create simple datasets for training and testing directly in the UI.
*   **Model Persistence:** Save trained models (architecture and weights) to a JSON file and load them back later.
*   **Visualization:**
    *   Real-time loss graph (Training and Validation).
    *   Network graph showing structure, node activations, and connection weights/types.
    *   Training history playback slider to step through epochs.
*   **Manual Prediction:** Test the trained model with custom inputs.

## How to Use

Clone or download this repository (or at least the `index.html` file and accompanying `src` folder), then open `index.html` in a modern web browser. No installation or build steps are required.

## Technology

Oblix is implemented in pure JavaScript with no external library dependencies for the core neural network logic. The `index.html` entry point imports modules from the `src` directory where all functionality resides.

## Benchmarks

The `benchmark/` directory contains small scripts that measure the speed of the
utility helpers. Benchmarks are provided for the data generation routines
`generateXORData`, `generateLinearData`, `generateCircularData` and
`generateGaussianBlobs`. Additional scripts cover `gaussianRandom`,
`positionalEncoding`, `calculateAccuracy`, `calculateRSquared` and
`softmaxForward`, and `dropoutForward`.

Run all performance benchmarks with:

```bash
node benchmark/run.js
```

## License

Released under the [MIT License](LICENSE).
