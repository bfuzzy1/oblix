# oblix API Documentation

This document provides comprehensive API documentation for the oblix neural network playground.

## Table of Contents
- [Core Classes](#core-classes)
- [Layer Operations](#layer-operations)
- [Activation Functions](#activation-functions)
- [Optimizers](#optimizers)
- [Utility Functions](#utility-functions)
- [Data Generation](#data-generation)
- [Training Configuration](#training-configuration)

## Core Classes

### Oblix Class (`src/network.js`)

The main neural network class that handles network construction, training, and inference.

#### Constructor
```javascript
new Oblix(debug = true)
```
- **debug** (boolean): Enable debug logging (default: true)

#### Methods

##### `reset()`
Resets the network to initial state, clearing all layers, weights, and training state.

##### `layer(config)`
Adds a layer to the network.

**Parameters:**
- **config** (object): Layer configuration
  - **type** (string): Layer type ('dense', 'layernorm', 'attention', 'dropout', 'softmax')
  - **inputSize** (number): Number of input features
  - **outputSize** (number): Number of output features
  - **activation** (string): Activation function (for dense layers)
  - **numHeads** (number): Number of attention heads (for attention layers)
  - **useBias** (boolean): Whether to use bias terms (for dense layers)
  - **rate** (number): Dropout rate (for dropout layers)
  - **weightInit** (string): Weight initialization method ('glorot', 'he', 'xavier')

**Returns:** Oblix instance (for chaining)

##### `initializeOptimizerState(optimizer)`
Initializes optimizer-specific state variables.

**Parameters:**
- **optimizer** (string): Optimizer type ('sgd', 'adam', 'rmsprop', 'adamw')

##### `getTotalParameters()`
Returns the total number of trainable parameters in the network.

**Returns:** number

##### `getCurrentLearningRate(epoch, initialLR, options)`
Calculates the current learning rate based on the schedule.

**Parameters:**
- **epoch** (number): Current training epoch
- **initialLR** (number): Initial learning rate
- **options** (object): Learning rate schedule options

**Returns:** number

##### `async train(trainSet, options = {})`
Trains the network on the provided dataset.

**Parameters:**
- **trainSet** (array): Training data array of [input, target] pairs
- **options** (object): Training configuration
  - **epochs** (number): Number of training epochs
  - **learningRate** (number): Initial learning rate
  - **optimizer** (string): Optimizer type
  - **lossFunction** (string): Loss function type
  - **batchSize** (number): Batch size for training
  - **l2Lambda** (number): L2 regularization strength
  - **gradientClipValue** (number): Gradient clipping threshold
  - **lrScheduler** (string): Learning rate scheduler type
  - **earlyStoppingPatience** (number): Early stopping patience
  - **printEveryEpochs** (number): Print progress every N epochs

**Returns:** Promise with training summary

##### `pauseTraining()`
Pauses the current training session.

##### `resumeTraining()`
Resumes a paused training session.

##### `predict(input)`
Performs forward pass to get predictions.

**Parameters:**
- **input** (Float32Array|array): Input data

**Returns:** Float32Array of predictions

##### `save(name = "model")`
Saves the trained model to a JSON file.

**Parameters:**
- **name** (string): Model name for the file

##### `load(callback)`
Loads a saved model from a JSON file.

**Parameters:**
- **callback** (function): Callback function to handle loaded model

## Layer Operations (`src/layers.js`)

### `oblixLayerOps`

#### `attentionForward(context, input, numHeads = 2)`
Performs multi-head self-attention forward pass.

**Parameters:**
- **context** (object): Network context with debug and forwardCache
- **input** (Float32Array): Input tensor
- **numHeads** (number): Number of attention heads

**Returns:** Float32Array of attention output

#### `attentionBackward(context, dOutput, cache)`
Performs multi-head self-attention backward pass.

**Parameters:**
- **context** (object): Network context
- **dOutput** (Float32Array): Gradient of output
- **cache** (object): Cached intermediate values from forward pass

**Returns:** object with dInput gradient

#### `dropoutForward(context, input, rate)`
Applies dropout during forward pass.

**Parameters:**
- **context** (object): Network context with isTraining flag
- **input** (Float32Array): Input tensor
- **rate** (number): Dropout rate (0-1)

**Returns:** Float32Array with dropout applied

#### `dropoutBackward(context, dOutput, cache)`
Performs dropout backward pass.

**Parameters:**
- **context** (object): Network context
- **dOutput** (Float32Array): Gradient of output
- **cache** (object): Cached dropout mask from forward pass

**Returns:** object with dInput gradient

#### `layerNormForward(context, input, gamma, beta)`
Performs layer normalization forward pass.

**Parameters:**
- **context** (object): Network context
- **input** (Float32Array): Input tensor
- **gamma** (Float32Array): Scale parameters
- **beta** (Float32Array): Shift parameters

**Returns:** object with output and cached values

#### `layerNormBackward(context, dOutput, cache)`
Performs layer normalization backward pass.

**Parameters:**
- **context** (object): Network context
- **dOutput** (Float32Array): Gradient of output
- **cache** (object): Cached values from forward pass

**Returns:** object with gradients

#### `softmaxForward(context, input)`
Performs softmax forward pass.

**Parameters:**
- **context** (object): Network context
- **input** (Float32Array): Input tensor

**Returns:** Float32Array of softmax probabilities

#### `softmaxBackward(context, dOutput, cache)`
Performs softmax backward pass.

**Parameters:**
- **context** (object): Network context
- **dOutput** (Float32Array): Gradient of output
- **cache** (object): Cached values from forward pass

**Returns:** object with dInput gradient

## Activation Functions (`src/activations.js`)

### `oblixActivations`

#### `apply(x, activation)`
Applies activation function to input.

**Parameters:**
- **x** (number): Input value
- **activation** (string): Activation function name

**Supported Activations:**
- `tanh`: Hyperbolic tangent
- `sigmoid`: Sigmoid function
- `relu`: Rectified Linear Unit
- `leakyrelu`: Leaky ReLU
- `gelu`: Gaussian Error Linear Unit
- `selu`: Scaled Exponential Linear Unit
- `swish`: Swish activation
- `mish`: Mish activation
- `softmax`: Softmax (identity in apply)
- `none`: Identity function

**Returns:** number

#### `derivative(x, activation)`
Calculates derivative of activation function.

**Parameters:**
- **x** (number): Input value
- **activation** (string): Activation function name

**Returns:** number

## Optimizers (`src/optimizers.js`)

### `oblixOptimizers`

#### `initializeState(context, optimizer)`
Initializes optimizer state variables.

**Parameters:**
- **context** (object): Network context
- **optimizer** (string): Optimizer type

#### `update(context, optimizer, learningRate, options = {})`
Performs parameter update using specified optimizer.

**Parameters:**
- **context** (object): Network context with gradients and state
- **optimizer** (string): Optimizer type ('sgd', 'adam', 'rmsprop', 'adamw')
- **learningRate** (number): Learning rate
- **options** (object): Optimizer-specific options
  - **beta1** (number): Adam beta1 parameter
  - **beta2** (number): Adam beta2 parameter
  - **epsilon** (number): Numerical stability constant
  - **decayRate** (number): RMSprop decay rate
  - **weightDecay** (number): Weight decay coefficient

## Utility Functions (`src/utils.js`)

### `oblixUtils`

#### `positionalEncoding(input, maxLen = -1)`
Adds positional encoding to input tensor.

**Parameters:**
- **input** (Float32Array): Input tensor
- **maxLen** (number): Maximum sequence length (-1 for auto)

**Returns:** Float32Array with positional encoding added

#### `calculateAccuracy(predictions, targets)`
Calculates classification accuracy.

**Parameters:**
- **predictions** (array): Array of prediction vectors
- **targets** (array): Array of target labels or one-hot vectors

**Returns:** number (accuracy percentage)

#### `calculateRSquared(predictions, targets)`
Calculates R-squared coefficient of determination.

**Parameters:**
- **predictions** (array): Array of predictions
- **targets** (array): Array of target values

**Returns:** number (R-squared value)

#### `gaussianRandom(mean = 0, std = 1)`
Generates random number from Gaussian distribution.

**Parameters:**
- **mean** (number): Mean of distribution
- **std** (number): Standard deviation

**Returns:** number

#### `fillRandomInts(array, min, max)`
Fills array with random integers.

**Parameters:**
- **array** (Uint32Array): Array to fill
- **min** (number): Minimum value (inclusive)
- **max** (number): Maximum value (exclusive)

## Data Generation Functions

### `generateXORData(numSamples, noiseLevel = 0.05)`
Generates XOR pattern dataset.

**Parameters:**
- **numSamples** (number): Number of samples to generate
- **noiseLevel** (number): Noise level (0-1)

**Returns:** array of [input, target] pairs

### `generateLinearData(numSamples, numInputs, numOutputs, noiseLevel = 0.05)`
Generates linear relationship dataset.

**Parameters:**
- **numSamples** (number): Number of samples
- **numInputs** (number): Number of input features
- **numOutputs** (number): Number of output values
- **noiseLevel** (number): Noise level (0-1)

**Returns:** array of [input, target] pairs

### `generateCircularData(numSamples, noiseLevel = 0.05)`
Generates circular pattern dataset.

**Parameters:**
- **numSamples** (number): Number of samples
- **noiseLevel** (number): Noise level (0-1)

**Returns:** array of [input, target] pairs

### `generateGaussianBlobs(numSamples, numClasses, noiseLevel = 0.05)`
Generates multi-class Gaussian blobs dataset.

**Parameters:**
- **numSamples** (number): Number of samples per class
- **numClasses** (number): Number of classes
- **noiseLevel** (number): Noise level (0-1)

**Returns:** array of [input, target] pairs

### `generateRandomData(numSamples, numInputs, numOutputs, noiseLevel = 0.05)`
Generates random dataset with configurable dimensions.

**Parameters:**
- **numSamples** (number): Number of samples
- **numInputs** (number): Number of input features
- **numOutputs** (number): Number of output values
- **noiseLevel** (number): Noise level (0-1)

**Returns:** array of [input, target] pairs

## Training Configuration

### Optimizer Options
- **SGD:** Stochastic Gradient Descent
- **Adam:** Adaptive Moment Estimation
- **RMSprop:** Root Mean Square Propagation
- **AdamW:** Adam with Weight Decay

### Loss Functions
- **MSE:** Mean Squared Error (regression)
- **Cross-Entropy:** Cross-entropy loss (classification)

### Learning Rate Schedulers
- **None:** Constant learning rate
- **Step Decay:** Reduce learning rate at specific epochs
- **Exponential Decay:** Continuous learning rate reduction

### Architecture Templates
- **Simple MLP:** Basic multi-layer perceptron
- **Basic Autoencoder:** Encoder-decoder architecture
- **Transformer Encoder Block:** Self-attention based
- **Residual Attention Block:** Attention with skip connections
- **MLP with Dropout:** Regularized multi-layer perceptron
- **Deep Residual MLP:** Deep network with residual connections
- **Transformer Stack:** Multiple transformer blocks
- **Autoencoder with Dropout:** Regularized autoencoder
- **Softmax Classifier:** Classification network

## Error Handling

All functions include comprehensive error handling:
- **Input validation** for parameters
- **Type checking** for arrays and objects
- **Graceful degradation** for unsupported operations
- **Meaningful error messages** for debugging

## Performance Notes

- **Float32Array** used for all numerical computations
- **Typed arrays** provide better performance than standard arrays
- **Crypto API** used for secure random number generation
- **Canvas API** used for real-time visualizations
- **ES6 modules** provide efficient code organization

## Browser Compatibility

- **Modern browsers** with ES6 support
- **Canvas API** for visualizations
- **File API** for model save/load
- **Web Crypto API** for random generation
- **Typed Arrays** for numerical operations