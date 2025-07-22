import { oblixActivations } from '../activations.js';
import { oblixLayerOps } from '../layers.js';
import { oblixOptimizers } from '../optimizers.js';
import { oblixUtils } from '../utils.js';
import { optimizedMath } from './math.js';

class OptimizedOblix {
  constructor(debug = true) {
    this.layers = [];
    this.weights = [];
    this.biases = [];
    this.gammas = [];
    this.betas = [];
    this.masks = [];
    
    this.details = {};
    this.debug = debug;
    this.usePositionalEncoding = false;
    this.isTraining = false;
    this.isPaused = false;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;
    this.t = 0;
    
    // Optimizer state
    this.m_dw = [];
    this.v_dw = [];
    this.m_db = [];
    this.v_db = [];
    this.m_dgamma = [];
    this.v_dgamma = [];
    this.m_dbeta = [];
    this.v_dbeta = [];
    this.s_dw = [];
    this.s_db = [];
    this.s_dgamma = [];
    this.s_dbeta = [];
    
    this.decayRate = 0.9;
    this.lastActivations = null;
    this.forwardCache = null;
    this.lastTrainLoss = null;
    
    if (this.debug) {
      console.log('Optimized Oblix instance created with performance optimizations');
    }
  }

  reset() {
    if (this.debug) console.log('Resetting optimized oblix instance...');
    
    this.layers = [];
    this.weights = [];
    this.biases = [];
    this.gammas = [];
    this.betas = [];
    this.masks = [];
    this.details = {};
    this.isTraining = false;
    this.isPaused = false;
    this.t = 0;
    
    // Reset optimizer state
    this.m_dw = [];
    this.v_dw = [];
    this.m_db = [];
    this.v_db = [];
    this.m_dgamma = [];
    this.v_dgamma = [];
    this.m_dbeta = [];
    this.v_dbeta = [];
    this.s_dw = [];
    this.s_db = [];
    this.s_dgamma = [];
    this.s_dbeta = [];
    
    this.lastActivations = null;
    this.forwardCache = null;
    this.lastTrainLoss = null;
    
    if (this.debug) console.log('Optimized Oblix reset complete.');
  }

  layer(config) {
    const {
      type = "dense",
      inputSize,
      outputSize,
      activation = "tanh",
      numHeads = 2,
      useBias = true,
      rate = 0.5,
      weightInit = "glorot",
    } = config;
    
    if (typeof inputSize !== "number" || inputSize <= 0) {
      throw new Error(`Layer ${this.layers.length}: Invalid inputSize: ${inputSize}.`);
    }
    
    if (this.layers.length > 0) {
      const prevLayer = this.layers[this.layers.length - 1];
      if (inputSize !== prevLayer.outputSize) {
        throw new Error(`Layer ${this.layers.length}: Input size mismatch. Expected ${prevLayer.outputSize}, got ${inputSize}.`);
      }
    }
    
    const layerConfig = {
      type,
      inputSize,
      outputSize,
      activation,
      numHeads,
      useBias,
      rate,
      weightInit,
    };
    
    this.layers.push(layerConfig);
    
    // Initialize weights and biases
    if (type === "dense") {
      const weightSize = inputSize * outputSize;
      const biasSize = useBias ? outputSize : 0;
      
      const weights = new Float32Array(weightSize);
      const biases = useBias ? new Float32Array(biasSize) : null;
      
      // Initialize weights using optimized methods
      this.initializeWeights(weights, weightInit, inputSize, outputSize);
      if (biases) {
        biases.fill(0);
      }
      
      this.weights.push(weights);
      this.biases.push(biases);
    } else {
      this.weights.push(null);
      this.biases.push(null);
    }
    
    // Initialize other layer-specific parameters
    this.gammas.push(null);
    this.betas.push(null);
    this.masks.push(null);
    
    return this;
  }

  initializeWeights(weights, method, inputSize, outputSize) {
    const fanIn = inputSize;
    const fanOut = outputSize;
    
    switch (method) {
      case "glorot":
        const scale = Math.sqrt(6 / (fanIn + fanOut));
        for (let i = 0; i < weights.length; i++) {
          weights[i] = (Math.random() - 0.5) * 2 * scale;
        }
        break;
      case "he":
        const heScale = Math.sqrt(2 / fanIn);
        for (let i = 0; i < weights.length; i++) {
          weights[i] = (Math.random() - 0.5) * 2 * heScale;
        }
        break;
      case "xavier":
        const xavierScale = Math.sqrt(1 / fanIn);
        for (let i = 0; i < weights.length; i++) {
          weights[i] = (Math.random() - 0.5) * 2 * xavierScale;
        }
        break;
      default:
        for (let i = 0; i < weights.length; i++) {
          weights[i] = (Math.random() - 0.5) * 2;
        }
    }
  }

  forward(input) {
    if (!(input instanceof Float32Array)) {
      input = new Float32Array(input);
    }
    
    const activations = [input];
    this.forwardCache = { activations: [input] };
    
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      const prevActivation = activations[activations.length - 1];
      
      let activation;
      
      switch (layer.type) {
        case "dense":
          activation = this.forwardDense(prevActivation, i);
          break;
        case "attention":
          activation = oblixLayerOps.attentionForward(this, prevActivation, layer.numHeads);
          break;
        case "layernorm":
          activation = oblixLayerOps.layerNormForward(this, prevActivation);
          break;
        case "dropout":
          activation = oblixLayerOps.dropoutForward(this, prevActivation, layer.rate);
          break;
        case "softmax":
          activation = optimizedMath.softmax(prevActivation);
          break;
        default:
          activation = prevActivation;
      }
      
      activations.push(activation);
      this.forwardCache.activations.push(activation);
    }
    
    this.lastActivations = activations;
    return activations[activations.length - 1];
  }

  forwardDense(input, layerIndex) {
    const layer = this.layers[layerIndex];
    const weights = this.weights[layerIndex];
    const biases = this.biases[layerIndex];
    
    // Use optimized matrix-vector multiplication
    const output = optimizedMath.matrixVectorMultiply(
      weights,
      input,
      layer.outputSize,
      layer.inputSize
    );
    
    // Add bias if present
    if (biases) {
      optimizedMath.vectorAdd(output, biases, output);
    }
    
    // Apply activation function using optimized batch operation
    if (layer.activation !== "none") {
      optimizedMath.batchActivation(output, layer.activation, output);
    }
    
    return output;
  }

  async train(trainSet, options = {}) {
    if (this.isTraining) {
      throw new Error("Already training");
    }
    
    this.isTraining = true;
    this.isPaused = false;
    
    const {
      epochs = 50,
      learningRate = 0.01,
      batchSize = 8,
      testSet = null,
      optimizer = "adam",
      lossFunction = "mse",
      l2Lambda = 0,
      decayRate = 0.9,
      gradientClipValue = 0,
      usePositionalEncoding = false,
      lrSchedule = "none",
      lrStepDecayFactor = 0.1,
      lrStepDecaySize = 10,
      lrExpDecayRate = 0.95,
      callback = null,
    } = options;
    
    // Initialize optimizer state
    this.initializeOptimizerState(optimizer);
    
    let trainLoss = 0;
    let testLoss = null;
    
    for (let epoch = 1; epoch <= epochs; epoch++) {
      if (this.isPaused) {
        await new Promise(resolve => {
          const checkResume = () => {
            if (!this.isPaused) resolve();
            else setTimeout(checkResume, 100);
          };
          checkResume();
        });
      }
      
      // Training epoch
      const epochLoss = await this.trainEpoch(trainSet, batchSize, learningRate, optimizer, lossFunction, l2Lambda, gradientClipValue);
      trainLoss = epochLoss;
      
      // Test epoch
      if (testSet && testSet.length > 0) {
        testLoss = await this.testEpoch(testSet, lossFunction);
      }
      
      // Callback
      if (callback) {
        await callback(epoch, trainLoss, testLoss, null, null, this.forwardCache);
      }
    }
    
    this.isTraining = false;
    
    return {
      trainLoss,
      testLoss,
      epochs,
    };
  }

  async trainEpoch(trainSet, batchSize, learningRate, optimizer, lossFunction, l2Lambda, gradientClipValue) {
    let totalLoss = 0;
    const numBatches = Math.ceil(trainSet.length / batchSize);
    
    for (let i = 0; i < numBatches; i++) {
      const start = i * batchSize;
      const end = Math.min(start + batchSize, trainSet.length);
      const batch = trainSet.slice(start, end);
      
      const batchLoss = await this.trainBatch(batch, learningRate, optimizer, lossFunction, l2Lambda, gradientClipValue);
      totalLoss += batchLoss;
    }
    
    return totalLoss / numBatches;
  }

  async trainBatch(batch, learningRate, optimizer, lossFunction, l2Lambda, gradientClipValue) {
    let totalLoss = 0;
    
    for (const sample of batch) {
      // Forward pass
      const prediction = this.forward(sample.input);
      
      // Compute loss
      const loss = this.computeLoss(prediction, sample.output, lossFunction);
      totalLoss += loss;
      
      // Backward pass (simplified for this example)
      // In a full implementation, you would compute gradients here
    }
    
    return totalLoss / batch.length;
  }

  async testEpoch(testSet, lossFunction) {
    let totalLoss = 0;
    
    for (const sample of testSet) {
      const prediction = this.forward(sample.input);
      const loss = this.computeLoss(prediction, sample.output, lossFunction);
      totalLoss += loss;
    }
    
    return totalLoss / testSet.length;
  }

  computeLoss(prediction, target, lossFunction) {
    switch (lossFunction) {
      case "mse":
        let mse = 0;
        for (let i = 0; i < prediction.length; i++) {
          const diff = prediction[i] - target[i];
          mse += diff * diff;
        }
        return mse / prediction.length;
      case "crossentropy":
        let ce = 0;
        for (let i = 0; i < prediction.length; i++) {
          ce -= target[i] * Math.log(Math.max(prediction[i], 1e-15));
        }
        return ce;
      default:
        return 0;
    }
  }

  initializeOptimizerState(optimizer) {
    // Initialize optimizer state arrays
    for (let i = 0; i < this.weights.length; i++) {
      if (this.weights[i]) {
        const size = this.weights[i].length;
        this.m_dw[i] = new Float32Array(size);
        this.v_dw[i] = new Float32Array(size);
        this.s_dw[i] = new Float32Array(size);
      }
      
      if (this.biases[i]) {
        const size = this.biases[i].length;
        this.m_db[i] = new Float32Array(size);
        this.v_db[i] = new Float32Array(size);
        this.s_db[i] = new Float32Array(size);
      }
    }
  }

  pauseTraining() {
    this.isPaused = true;
  }

  resumeTraining() {
    this.isPaused = false;
  }

  predict(input) {
    return this.forward(input);
  }

  getTotalParameters() {
    let total = 0;
    for (const weights of this.weights) {
      if (weights) total += weights.length;
    }
    for (const biases of this.biases) {
      if (biases) total += biases.length;
    }
    return total;
  }

  getCurrentLearningRate(epoch, initialLR, options) {
    const { lrSchedule, lrStepDecayFactor, lrStepDecaySize, lrExpDecayRate } = options;
    
    switch (lrSchedule) {
      case "step":
        return initialLR * Math.pow(lrStepDecayFactor, Math.floor(epoch / lrStepDecaySize));
      case "exponential":
        return initialLR * Math.pow(lrExpDecayRate, epoch);
      default:
        return initialLR;
    }
  }
}

export { OptimizedOblix };