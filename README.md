# oblix

**oblix** is a self-contained, browser-based neural network playground written entirely in pure JavaScript. It provides an interactive environment to build, train, visualize, and experiment with various neural network architectures and training parameters. The main HTML page loads JavaScript modules from the `src` directory.

## 🚀 Features

### Core Neural Network Capabilities
- **Interactive UI:** Build networks and configure training through a graphical interface
- **Multiple Layer Types:** 
  - **Dense (Fully Connected):** Standard neural network layers with configurable activation functions
  - **Layer Normalization:** Normalize activations across features for stable training
  - **Self-Attention:** Multi-head attention mechanism for sequence modeling
  - **Dropout:** Regularization technique to prevent overfitting
  - **Softmax:** Output layer for classification tasks

### Architecture Templates
Quickly load predefined structures:
- **Simple MLP:** Basic multi-layer perceptron
- **Basic Autoencoder:** Encoder-decoder architecture
- **Transformer Encoder Block:** Self-attention based architecture
- **Residual Attention Block:** Attention with skip connections
- **MLP with Dropout:** Regularized multi-layer perceptron
- **Deep Residual MLP:** Deep network with residual connections
- **Transformer Stack:** Multiple transformer blocks
- **Autoencoder with Dropout:** Regularized autoencoder
- **Softmax Classifier:** Classification network

### Training Configuration
- **Optimizers:** SGD, Adam, RMSprop, AdamW
- **Loss Functions:** Mean Squared Error (MSE), Cross-Entropy
- **Learning Rate:** Set initial rate with optional schedules:
  - **Step Decay:** Reduce learning rate at specific epochs
  - **Exponential Decay:** Continuous learning rate reduction
- **Regularization:** L2 Weight Decay for overfitting prevention
- **Gradient Clipping:** Prevent exploding gradients
- **Batch Size:** Control the number of samples per update

### Advanced Features
- **Positional Encoding:** Add positional information to inputs for sequence-like data
- **Built-in Data Generation:** Create synthetic datasets directly in the UI:
  - XOR patterns
  - Linear relationships
  - Circular patterns
  - Gaussian blobs (multi-class)
  - Random data with configurable noise
- **Model Persistence:** Save trained models (architecture and weights) to JSON and load them back
- **CSV Import/Export:** Load training data from CSV files and export predictions

### Visualization & Analysis
- **Real-time Loss Graph:** Training and validation loss curves
- **Network Graph:** Interactive visualization showing:
  - Network structure and connections
  - Node activations during forward pass
  - Connection weights and types
  - Layer-by-layer information flow
- **Training History Playback:** Step through epochs with a slider
- **Manual Prediction:** Test trained models with custom inputs
- **Performance Metrics:** Accuracy, R-squared, and loss tracking

## 🛠️ Technology Stack

### Core Implementation
- **Pure JavaScript:** No external library dependencies for neural network logic
- **ES6 Modules:** Modern JavaScript module system
- **Float32Array:** Optimized typed arrays for numerical computations
- **Canvas API:** Real-time visualization and graphing
- **Web Crypto API:** Secure random number generation for dropout masks

### Performance Optimizations
- **Typed Arrays:** Float32Array for efficient memory usage
- **Optimized Approximations:** Fast implementations of GELU activation and gaussian random generation
- **Crypto-based Randomness:** `crypto.getRandomValues` for dropout masks
- **Deterministic Testing:** Custom `randomFillFn` for reproducible results

### Browser Compatibility
- **Modern Browsers:** Chrome, Firefox, Safari, Edge
- **ES6 Support:** Arrow functions, destructuring, modules
- **Canvas Support:** 2D graphics for visualizations
- **File API:** Model save/load functionality

## 📖 How to Use

### Quick Start
1. **Clone or download** this repository
2. **Open `index.html`** in a modern web browser
3. **No installation or build steps required**

### Basic Workflow
1. **Configure Network Architecture:**
   - Select number of hidden layers
   - Choose layer types and activation functions
   - Set input/output dimensions

2. **Prepare Training Data:**
   - Use built-in data generators (XOR, linear, circular, blobs)
   - Import CSV files with custom data
   - Configure data parameters

3. **Set Training Parameters:**
   - Choose optimizer (SGD, Adam, RMSprop, AdamW)
   - Select loss function (MSE, Cross-Entropy)
   - Configure learning rate and schedules
   - Set regularization parameters

4. **Train and Monitor:**
   - Start training with real-time loss visualization
   - Pause/resume training as needed
   - Monitor network graph and activations

5. **Evaluate and Save:**
   - Test model with custom inputs
   - Save trained model to JSON file
   - Load previously saved models

## 🧪 Testing & Benchmarks

### Running Tests
```bash
npm test
# or
node tests/run.js
```

### Running Benchmarks
```bash
npm run benchmark
# or
node benchmark/run.js
```

### Test Coverage
- **Layer Operations:** Attention, dropout, layer normalization, softmax
- **Utility Functions:** Accuracy calculation, R-squared, positional encoding
- **Data Generation:** XOR, linear, circular, gaussian blobs
- **Network Operations:** Learning rate scheduling, parameter counting
- **Optimizers:** All optimizer implementations
- **UI Elements:** Interface functionality

### Performance Benchmarks
- **Activation Functions:** ReLU, tanh, sigmoid, GELU, SELU, Swish, Mish
- **Layer Operations:** Attention, dropout, layer normalization, softmax
- **Data Generation:** Various synthetic dataset generators
- **Utility Functions:** Accuracy, R-squared, positional encoding

## 📁 Project Structure

```
oblix/
├── index.html              # Main entry point
├── src/                    # Core neural network implementation
│   ├── main.js            # UI and application logic
│   ├── network.js         # Main neural network class
│   ├── layers.js          # Layer operations (attention, dropout, etc.)
│   ├── activations.js     # Activation functions
│   ├── optimizers.js      # Optimizer implementations
│   └── utils.js           # Utility functions
├── tests/                  # Test suite
│   ├── run.js             # Test runner
│   └── test_*.js          # Individual test files
├── benchmark/              # Performance benchmarks
│   ├── run.js             # Benchmark runner
│   └── bench_*.js         # Individual benchmark files
├── README.md              # This file
├── AGENTS.md              # Development guidelines
├── ROADMAP.md             # Future development plans
└── LICENSE                # MIT License
```

## 🔧 Development

### Code Style
- **2-space indentation**
- **Semicolons required**
- **ES6 modules**
- **Pure JavaScript** (no external dependencies)

### Testing Requirements
- **Run tests after each code change:** `node tests/run.js`
- **Maintain test coverage** for all new features
- **Ensure all tests pass** before committing

### Adding Features
- **Keep UI simple and educational**
- **Maintain self-contained architecture**
- **Update documentation thoroughly**
- **Add corresponding tests**

## 📈 Performance Characteristics

### Activation Function Performance (1M operations)
- **ReLU:** ~5ms (fastest)
- **LeakyReLU:** ~5ms
- **SELU:** ~6ms
- **Swish:** ~7ms
- **GELU:** ~9ms
- **Sigmoid:** ~10ms
- **Tanh:** ~13ms
- **Mish:** ~16ms (slowest)

### Layer Operations
- **Attention (128-dim, 8 heads):** ~0.5ms
- **Dropout (10K elements):** ~0.7ms
- **Layer Normalization (256 dims):** ~5ms
- **Softmax (512 dims):** ~0.1ms

### Data Generation (10K samples)
- **XOR Data:** ~2ms
- **Circular Data:** ~2ms
- **Gaussian Blobs:** ~2ms
- **Linear Data:** ~8ms

## 🤝 Contributing

### Pull Request Format
**Context:** Explain the motivation and project perspective
**Description:** Detailed technical implementation steps
**Changes:** Specific code modifications and functionality added

### Development Guidelines
- **Maintain self-contained architecture**
- **Add comprehensive tests**
- **Update documentation**
- **Keep UI educational and simple**

## 📄 License

Released under the [MIT License](LICENSE).

## 🔮 Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plans including:
- **GitHub Pages hosting**
- **Enhanced documentation and tutorials**
- **CSV import/export improvements**
- **Local storage for model persistence**
- **Additional layer types (convolutional, recurrent)**
- **WebGPU acceleration**
- **Plugin system for community extensions**
