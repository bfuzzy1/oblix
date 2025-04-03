# oblix

**oblix** is a self-contained, browser-based neural network playground written entirely in pure JavaScript. It provides an interactive environment to build, train, visualize, and experiment with various neural network architectures and training parameters directly within a single HTML file.

## Features

*   **Interactive UI:** Build networks and configure training through a graphical interface.
*   **Multiple Layer Types:** Supports Dense (fully connected), Layer Normalization, Self-Attention, Dropout, and Softmax layers.
*   **Architecture Templates:** Quickly load predefined structures like Simple MLP, Basic Autoencoder, Transformer Encoder block, and Residual Attention block.
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
*   **Manual Prediction:** Test the trained model with custom inputs.

## How to Use

Simply download the `index.html` file and open it in a modern web browser. No installation or build steps are required.

## Technology

Oblix is implemented in pure JavaScript with no external library dependencies for the core neural network logic. All functionality is contained within the single `index.html` file.