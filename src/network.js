import { oblixActivations } from './activations.js';
import { oblixLayerOps } from './layers.js';
import { oblixOptimizers } from './optimizers.js';
import { oblixUtils } from './utils.js';

/**
 *
 */
class Oblix {
  /**
   *
   */
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

    if (this.debug)
      console.log(
        'oblix instance created. Initializing for Float32Array storage...'
      );
  }

  /**
   *
   */
  reset() {
    if (this.debug) console.log('Resetting oblix instance...');
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
    if (this.debug) console.log('Oblix reset complete.');
  }

  /**
   * Validates layer configuration parameters.
   * Business rule: Layer validation prevents runtime errors.
   *
   * @param {Object} config - Layer configuration
   * @param {string} config.type - Layer type
   * @param {number} config.inputSize - Input size
   * @param {number} config.outputSize - Output size
   * @param {number} config.numHeads - Number of attention heads
   * @param {number} config.rate - Dropout rate
   * @throws {Error} If configuration is invalid
   */
  validateLayerConfig(config) {
    const { type, inputSize, outputSize, numHeads, rate } = config;
    
    if (typeof inputSize !== 'number' || inputSize <= 0) {
      throw new Error(`Layer ${this.layers.length}: Invalid inputSize: ${inputSize}.`);
    }
    
    if (this.layers.length > 0) {
      const prevLayer = this.layers[this.layers.length - 1];
      if (inputSize !== prevLayer.outputSize) {
        throw new Error(
          `Layer ${this.layers.length} (${type}): Input size ${inputSize} doesn't match previous layer's output size ${prevLayer.outputSize}.`
        );
      }
    }
    
    if (type === 'dense') {
      if (typeof outputSize !== 'number' || outputSize <= 0) {
        throw new Error(`Dense Layer ${this.layers.length}: Invalid outputSize: ${outputSize}.`);
      }
    }
    
    if (type === 'attention' && inputSize % numHeads !== 0) {
      throw new Error(
        `Attention layer ${this.layers.length}: Input size ${inputSize} not divisible by numHeads ${numHeads}.`
      );
    }
    
    if (type === 'dropout' && (rate < 0 || rate >= 1)) {
      throw new Error(
        `Dropout layer ${this.layers.length}: Rate ${rate} must be >= 0 and < 1.`
      );
    }
  }

  /**
   * Determines the actual output size for a layer.
   * Business rule: Output size calculation depends on layer type.
   *
   * @param {Object} config - Layer configuration
   * @param {string} config.type - Layer type
   * @param {number} config.inputSize - Input size
   * @param {number} config.outputSize - Output size
   * @returns {number} Actual output size
   */
  determineOutputSize(config) {
    const { type, inputSize, outputSize } = config;
    
    switch (type) {
    case 'dense':
      return outputSize;
    case 'layernorm':
    case 'attention':
    case 'dropout':
    case 'softmax':
      if (outputSize !== undefined && outputSize !== inputSize) {
        console.warn(`${type} layer ${this.layers.length}: Output size ignored.`);
      }
      return inputSize;
    default:
      throw new Error(`Unknown layer type: ${type}`);
    }
  }

  /**
   * Creates layer configuration object.
   * Business rule: Layer configuration must be consistent.
   *
   * @param {Object} config - Layer configuration
   * @param {number} actualOutputSize - Calculated output size
   * @returns {Object} Layer configuration object
   */
  createLayerConfig(config, actualOutputSize) {
    const {
      type = 'dense',
      inputSize,
      activation = 'tanh',
      numHeads = 2,
      useBias = true,
      rate = 0.5,
      weightInit = 'glorot'
    } = config;
    
    return {
      type,
      inputSize,
      outputSize: actualOutputSize,
      activation,
      numHeads,
      useBias,
      rate,
      weightInit
    };
  }

  /**
   * Initializes optimizer state arrays for a layer.
   * Business rule: Optimizer state must be properly initialized.
   *
   * @param {number} layerIndex - Index of the layer
   */
  initializeOptimizerArrays(layerIndex) {
    this.weights.push(null);
    this.biases.push(null);
    this.gammas.push(null);
    this.betas.push(null);
    this.masks.push(null);
    this.m_dw.push(null);
    this.v_dw.push(null);
    this.m_db.push(null);
    this.v_db.push(null);
    this.m_dgamma.push(null);
    this.v_dgamma.push(null);
    this.m_dbeta.push(null);
    this.v_dbeta.push(null);
    this.s_dw.push(null);
    this.s_db.push(null);
    this.s_dgamma.push(null);
    this.s_dbeta.push(null);
  }

  /**
   * Initializes weights for a dense layer.
   * Business rule: Weight initialization affects training convergence.
   *
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Index of the layer
   */
  initializeDenseWeights(layerConfig, layerIndex) {
    const { inputSize, outputSize, weightInit } = layerConfig;
    const weightCount = outputSize * inputSize;
    const weightsArray = new Float32Array(weightCount);
    
    let initFunc;
    if (weightInit === 'he') {
      const stdDev = Math.sqrt(2 / inputSize);
      initFunc = () => oblixUtils.gaussianRandom() * stdDev;
      if (this.debug) {
        console.log(`L${layerIndex} Dense Weights init: He (stdDev=${stdDev.toFixed(4)})`);
      }
    } else {
      const limit = Math.sqrt(6 / (inputSize + outputSize));
      initFunc = () => (Math.random() * 2 - 1) * limit;
      if (this.debug) {
        console.log(`L${layerIndex} Dense Weights init: Glorot (limit=${limit.toFixed(4)})`);
      }
    }
    
    for (let i = 0; i < weightCount; i++) {
      weightsArray[i] = initFunc();
    }
    
    this.weights[layerIndex] = weightsArray;
    
    if (layerConfig.useBias) {
      const biasesArray = new Float32Array(outputSize);
      for (let i = 0; i < outputSize; i++) {
        biasesArray[i] = 0;
      }
      this.biases[layerIndex] = biasesArray;
    }
  }

  /**
   * Initializes parameters for non-dense layers.
   * Business rule: Different layer types require different parameter initialization.
   *
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Index of the layer
   */
  initializeNonDenseParameters(layerConfig, layerIndex) {
    const { type, inputSize, outputSize } = layerConfig;
    
    switch (type) {
    case 'layernorm':
      this.gammas[layerIndex] = new Float32Array(inputSize).fill(1);
      this.betas[layerIndex] = new Float32Array(inputSize).fill(0);
      break;
    case 'attention':
      // Attention layers use the same input/output size
      this.weights[layerIndex] = new Float32Array(inputSize * inputSize).fill(0);
      if (layerConfig.useBias) {
        this.biases[layerIndex] = new Float32Array(inputSize).fill(0);
      }
      break;
    case 'dropout':
      // Dropout layers don't need weights/biases
      this.masks[layerIndex] = null;
      break;
    case 'softmax':
      // Softmax layers don't need additional parameters
      break;
    }
  }

  /**
   * Adds a layer to the neural network.
   * Business rule: Layer addition must be validated and properly initialized.
   *
   * @param {Object} config - Layer configuration
   */
  layer(config) {
    this.validateLayerConfig(config);
    
    const actualOutputSize = this.determineOutputSize(config);
    const layerConfig = this.createLayerConfig(config, actualOutputSize);
    const layerIndex = this.layers.length;
    
    this.layers.push(layerConfig);
    this.initializeOptimizerArrays(layerIndex);
    
    if (layerConfig.type === 'dense') {
      this.initializeDenseWeights(layerConfig, layerIndex);
    } else {
      this.initializeNonDenseParameters(layerConfig, layerIndex);
    }
  }

  /**
   *
   */
  initializeOptimizerState(optimizer) {
    const numLayers = this.layers.length;
    if (this.debug)
      console.log(
        `Init optimizer state (${optimizer}) for ${numLayers} layers. Creating Float32Arrays.`
      );
    this.t = 0;

    const ensureLen = (arrName) => {
      if (!this[arrName] || this[arrName].length !== numLayers) {
        this[arrName] = Array(numLayers).fill(null);
      }
    };
    [
      'm_dw',
      'v_dw',
      'm_db',
      'v_db',
      'm_dgamma',
      'v_dgamma',
      'm_dbeta',
      'v_dbeta',
      's_dw',
      's_db',
      's_dgamma',
      's_dbeta'
    ].forEach(ensureLen);

    for (let i = 0; i < numLayers; i++) {
      const cfg = this.layers[i];
      if (!cfg) continue;
      const w = this.weights[i];
      const b = this.biases[i];
      const g = this.gammas[i];
      const beta = this.betas[i];

      const needsWState = cfg.type === 'dense' && w instanceof Float32Array;
      const needsBState =
        cfg.type === 'dense' && cfg.useBias && b instanceof Float32Array;
      const needsLNState =
        cfg.type === 'layernorm' &&
        g instanceof Float32Array &&
        beta instanceof Float32Array;

      if (
        optimizer === 'adam' ||
        optimizer === 'rmsprop' ||
        optimizer === 'adamw'
      ) {
        try {
          if (needsWState) {
            const size = w.length;
            if (optimizer === 'adam' || optimizer === 'adamw') {
              if (!this.m_dw[i]) this.m_dw[i] = new Float32Array(size).fill(0);
              if (!this.v_dw[i]) this.v_dw[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === 'rmsprop') {
              if (!this.s_dw[i]) this.s_dw[i] = new Float32Array(size).fill(0);
            }

            if (this.debug)
              console.log(
                `L${i} Dense W Opt State (${optimizer}) init: ${this.m_dw[i]?.length || this.s_dw[i]?.length} elements`
              );
          }
          if (needsBState) {
            const size = b.length;
            if (optimizer === 'adam' || optimizer === 'adamw') {
              if (!this.m_db[i]) this.m_db[i] = new Float32Array(size).fill(0);
              if (!this.v_db[i]) this.v_db[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === 'rmsprop') {
              if (!this.s_db[i]) this.s_db[i] = new Float32Array(size).fill(0);
            }
            if (this.debug)
              console.log(
                `L${i} Dense B Opt State (${optimizer}) init: ${this.m_db[i]?.length || this.s_db[i]?.length} elements`
              );
          }
          if (needsLNState) {
            const size = g.length;
            if (optimizer === 'adam' || optimizer === 'adamw') {
              if (!this.m_dgamma[i])
                this.m_dgamma[i] = new Float32Array(size).fill(0);
              if (!this.v_dgamma[i])
                this.v_dgamma[i] = new Float32Array(size).fill(0);
              if (!this.m_dbeta[i])
                this.m_dbeta[i] = new Float32Array(size).fill(0);
              if (!this.v_dbeta[i])
                this.v_dbeta[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === 'rmsprop') {
              if (!this.s_dgamma[i])
                this.s_dgamma[i] = new Float32Array(size).fill(0);
              if (!this.s_dbeta[i])
                this.s_dbeta[i] = new Float32Array(size).fill(0);
            }
            if (this.debug)
              console.log(
                `L${i} LN Opt State (${optimizer}) init: ${this.m_dgamma[i]?.length || this.s_dgamma[i]?.length} elements`
              );
          }
        } catch (e) {
          console.error(`InitOpt State Err L${i} (${optimizer}): ${e.message}`);

          this.m_dw[i] = null;
          this.v_dw[i] = null;
          this.s_dw[i] = null;
          this.m_db[i] = null;
          this.v_db[i] = null;
          this.s_db[i] = null;
          this.m_dgamma[i] = null;
          this.v_dgamma[i] = null;
          this.s_dgamma[i] = null;
          this.m_dbeta[i] = null;
          this.v_dbeta[i] = null;
          this.s_dbeta[i] = null;
        }
      }
    }
    if (this.debug) console.log('Optimizer state init finished.');
  }

  /**
   *
   */
  getTotalParameters() {
    let total = 0;
    if (!this.layers || this.layers.length === 0) return 0;
    this.layers.forEach((l, i) => {
      if (l.type === 'dense') {
        total += this.weights[i] ? this.weights[i].length : 0;
        total += l.useBias && this.biases[i] ? this.biases[i].length : 0;
      } else if (l.type === 'layernorm') {
        total += this.gammas[i] ? this.gammas[i].length : 0;
        total += this.betas[i] ? this.betas[i].length : 0;
      }
    });

    if (this.debug)
      console.log(` getTotalParameters: Calculated total: ${total}`);
    return total;
  }

  /**
   *
   */
  getCurrentLearningRate(epoch, initialLR, options) {
    const { lrSchedule, lrStepDecayFactor, lrStepDecaySize, lrExpDecayRate } =
      options;
    let currentLR = initialLR;

    if (lrSchedule === 'step') {
      const stepSize = Math.max(1, Math.floor(lrStepDecaySize));

      const decaySteps = Math.floor(epoch / stepSize);
      currentLR = initialLR * Math.pow(lrStepDecayFactor, decaySteps);
    } else if (lrSchedule === 'exponential') {
      currentLR = initialLR * Math.pow(lrExpDecayRate, epoch);
    }

    return currentLR;
  }

  async train(trainSet, options = {}) {
    this.isTraining = true;
    const start = Date.now();
    
    const trainingConfig = this.initializeTrainingConfig(options);
    this.validateTrainingSetup(trainSet);
    
    const effectiveBatchSize = Math.max(
      1,
      Math.min(trainingConfig.batchSize, trainSet.length)
    );
    
    this.setupPositionalEncoding(trainingConfig);
    this.initializeOptimizerIfNeeded(trainingConfig.optimizer);
    
    const trainingState = this.initializeTrainingState();
    
    for (let epoch = 0; epoch < trainingConfig.epochs; epoch++) {
      await this.handleTrainingPause();
      
      const currentEpochLearningRate = this.getCurrentLearningRate(
        epoch,
        trainingConfig.initialLearningRate,
        trainingConfig
      );

      const epochResult = await this.trainEpoch(
        trainSet,
        effectiveBatchSize,
        currentEpochLearningRate,
        trainingConfig,
        trainingState
      );
      
      await this.handleEpochCallback(
        epoch,
        epochResult,
        trainingConfig,
        trainingState
      );
      
      if (this.shouldEarlyStop(trainingState, trainingConfig)) {
        break;
      }
    }
    
    this.isTraining = false;
    return this.createTrainingSummary(trainingState, start);
  }

  initializeTrainingConfig(options) {
    const {
      epochs = 100,
      learningRate = 0.01,
      batchSize = 16,
      printEveryEpochs = 10,
      earlyStopThreshold = 1e-7,
      testSet = null,
      callback = null,
      optimizer = 'adam',
      lossFunction = 'mse',
      l2Lambda = 0,
      decayRate = 0.9,
      usePositionalEncoding = this.usePositionalEncoding,
      gradientClipValue = 0,
      lrSchedule = 'none',
      lrStepDecayFactor = 0.1,
      lrStepDecaySize = 10,
      lrExpDecayRate = 0.95
    } = options;

    return {
      epochs,
      learningRate,
      initialLearningRate: learningRate,
      batchSize,
      printEveryEpochs,
      earlyStopThreshold,
      testSet,
      callback,
      optimizer,
      lossFunction,
      l2Lambda,
      decayRate,
      usePositionalEncoding,
      gradientClipValue,
      lrSchedule,
      lrStepDecayFactor,
      lrStepDecaySize,
      lrExpDecayRate
    };
  }

  validateTrainingSetup(trainSet) {
    if (!trainSet || trainSet.length === 0) {
      throw new Error('Training set empty.');
    }
    if (this.layers.length === 0) {
      throw new Error('No layers.');
    }
  }

  setupPositionalEncoding(config) {
    this.usePositionalEncoding = config.usePositionalEncoding;
    this.decayRate = config.decayRate;
  }

  initializeOptimizerIfNeeded(optimizer) {
    let needsOptimizerInit =
      this.m_dw?.length !== this.layers.length ||
      this.v_dw?.length !== this.layers.length ||
      this.s_dw?.length !== this.layers.length ||
      this.m_db?.length !== this.layers.length ||
      this.v_db?.length !== this.layers.length ||
      this.s_db?.length !== this.layers.length;
      
    for (let i = 0; i < this.layers.length && !needsOptimizerInit; i++) {
      if (this.layers[i].type === 'dense') {
        if (this.weights[i] && this.m_dw[i] === null) needsOptimizerInit = true;
        if (this.biases[i] && this.m_db[i] === null) needsOptimizerInit = true;
      } else if (this.layers[i].type === 'layernorm') {
        if (this.gammas[i] && this.m_dgamma[i] === null)
          needsOptimizerInit = true;
        if (this.betas[i] && this.m_dbeta[i] === null)
          needsOptimizerInit = true;
      }
    }
    
    if (needsOptimizerInit) {
      if (this.debug) console.log('Optimizer state needs init.');
      this.initializeOptimizerState(optimizer);
    }
  }

  initializeTrainingState() {
    return {
      lastTrainLoss: Infinity,
      lastTestLoss: null,
      lastValidationMetric: null,
      validationMetricName: ''
    };
  }

  async handleTrainingPause() {
    while (this.isPaused) {
      await new Promise((r) => setTimeout(r, 50));
    }
  }

  async trainEpoch(trainSet, effectiveBatchSize, currentEpochLearningRate, config, state) {
    this.shuffleTrainingData(trainSet);
    
    let totalEpochTrainError = 0;
    
    for (let b = 0; b < trainSet.length; b += effectiveBatchSize) {
      await this.handleTrainingPause();
      
      const batch = trainSet.slice(b, b + effectiveBatchSize);
      if (batch.length === 0) continue;

      const batchResult = await this.trainBatch(
        batch,
        currentEpochLearningRate,
        config
      );
      
      totalEpochTrainError += batchResult.loss;
    }
    
    const averageTrainLoss = totalEpochTrainError / trainSet.length;
    state.lastTrainLoss = averageTrainLoss;
    
    return { averageTrainLoss };
  }

  shuffleTrainingData(trainSet) {
    for (let i = trainSet.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]];
    }
  }

  async trainBatch(batch, learningRate, config) {
    const gradients = this.initializeGradients();
    let batchLossSum = 0;
    
    for (const data of batch) {
      await this.handleTrainingPause();
      
      const forwardResult = this.forward(data.input);
      const loss = this.calculateLoss(forwardResult, data.output, config);
      batchLossSum += loss;
      
      this.backward(data.output, config);
      this.accumulateGradients(gradients);
    }
    
    this.updateParameters(gradients, learningRate, config);
    
    return { loss: batchLossSum };
  }

  initializeGradients() {
    return {
      weights: this.weights.map((L) =>
        L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null
      ),
      biases: this.biases.map((L) =>
        L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null
      ),
      gammas: this.gammas.map((L) =>
        L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null
      ),
      betas: this.betas.map((L) =>
        L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null
      )
    };
  }

  calculateLoss(prediction, target, config) {
    switch (config.lossFunction) {
      case 'mse':
        return this.calculateMSELoss(prediction, target);
      case 'crossentropy':
        return this.calculateCrossEntropyLoss(prediction, target, this.layers[this.layers.length - 1]);
      default:
        throw new Error(`Unknown loss function: ${config.lossFunction}`);
    }
  }

  calculateMSELoss(prediction, target) {
    let loss = 0;
    for (let i = 0; i < prediction.length; i++) {
      const diff = prediction[i] - target[i];
      loss += diff * diff;
    }
    return loss / prediction.length;
  }



  async handleEpochCallback(epoch, epochResult, config, state) {
    const testLoss = await this.calculateTestLoss(config);
    state.lastTestLoss = testLoss;
    
    const validationMetric = await this.calculateValidationMetric(config, state);
    
    if (config.callback) {
      await config.callback(
        epoch + 1,
        epochResult.averageTrainLoss,
        testLoss,
        state.validationMetricName,
        validationMetric,
        this.lastForwardCache
      );
    }
  }

  async calculateTestLoss(config) {
    if (!config.testSet || config.testSet.length === 0) {
      return null;
    }
    
    let totalTestLoss = 0;
    for (const data of config.testSet) {
      const prediction = this.forward(data.input);
      const loss = this.calculateLoss(prediction, data.output, config);
      totalTestLoss += loss;
    }
    
    return totalTestLoss / config.testSet.length;
  }

  async calculateValidationMetric(config, state) {
    if (!config.testSet || config.testSet.length === 0) {
      return null;
    }
    
    const predictions = [];
    const targets = [];
    
    for (const data of config.testSet) {
      const prediction = this.forward(data.input);
      predictions.push(prediction);
      targets.push(data.output);
    }
    
    if (config.lossFunction === 'crossentropy') {
      const accuracy = this.calculateAccuracy(predictions, targets);
      state.validationMetricName = 'Accuracy';
      return accuracy;
    } else {
      const rSquared = this.calculateRSquared(predictions, targets);
      state.validationMetricName = 'R²';
      return rSquared;
    }
  }

  calculateAccuracy(predictions, targets) {
    let correct = 0;
    let total = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      const predIndex = this.getMaxIndex(predictions[i]);
      const targetIndex = this.getMaxIndex(targets[i]);
      if (predIndex === targetIndex) {
        correct++;
      }
      total++;
    }
    
    return total > 0 ? correct / total : 0;
  }

  getMaxIndex(array) {
    let maxIndex = 0;
    let maxValue = array[0];
    
    for (let i = 1; i < array.length; i++) {
      if (array[i] > maxValue) {
        maxValue = array[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  calculateRSquared(predictions, targets) {
    const allPredictions = predictions.flat();
    const allTargets = targets.flat();
    
    const mean = allTargets.reduce((sum, val) => sum + val, 0) / allTargets.length;
    const ssRes = allPredictions.reduce((sum, pred, i) => {
      const diff = pred - allTargets[i];
      return sum + diff * diff;
    }, 0);
    
    const ssTot = allTargets.reduce((sum, target) => {
      const diff = target - mean;
      return sum + diff * diff;
    }, 0);
    
    return ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
  }

  shouldEarlyStop(state, config) {
    if (config.earlyStopThreshold <= 0) {
      return false;
    }
    
    const lossDiff = Math.abs(state.lastTrainLoss - state.lastTrainLoss);
    return lossDiff < config.earlyStopThreshold;
  }

  createTrainingSummary(state, start) {
    const end = Date.now();
    const trainingTime = end - start;
    
    return {
      trainLoss: state.lastTrainLoss,
      testLoss: state.lastTestLoss,
      trainingTime,
      epochs: this.currentEpoch || 0
    };
  }

  /**
   *
   */
  pauseTraining() {
    this.isPaused = true;
  }

  /**
   *
   */
  resumeTraining() {
    this.isPaused = false;
  }

  /**
   *
   */
  predict(input) {
    if (this.debug) {
      console.log(' Starting prediction with native Float32Array logic.');
    }

    const wasTraining = this.isTraining;
    this.isTraining = false;
    
    this.validatePredictionInput(input);
    
    let currentInput = this.prepareInputForPrediction(input);
    
    if (this.usePositionalEncoding) {
      currentInput = this.applyPositionalEncoding(currentInput);
    }

    this.lastActivations = [currentInput];

    try {
      const output = this.processLayersForPrediction();
      return output;
    } catch (error) {
      console.error('Prediction error:', error);
      return null;
    } finally {
      this.isTraining = wasTraining;
    }
  }

  validatePredictionInput(input) {
    if (!this.layers || this.layers.length === 0) {
      throw new Error('Predict Error: Model not initialized.');
    }
    if (!input || typeof input.length !== 'number') {
      throw new Error('Predict Error: Invalid input provided.');
    }
  }

  prepareInputForPrediction(input) {
    if (input instanceof Float32Array) {
      return input;
    } else if (Array.isArray(input)) {
      if (this.debug) {
        console.log(' Input is standard array, converting to Float32Array.');
      }
      return new Float32Array(input);
    } else {
      throw new Error('Predict Error: Input is not an array or Float32Array.');
    }
  }

  applyPositionalEncoding(input) {
    if (this.debug) {
      console.log(' Applying positional encoding.');
    }
    const encodedInput = oblixUtils.positionalEncoding(input);
    if (this.debug) {
      console.log(
        ` After PosEnc type=${encodedInput.constructor.name}, len=${encodedInput.length}`
      );
    }
    return encodedInput;
  }

  processLayersForPrediction() {
    for (let i = 0; i < this.layers.length; i++) {
      const cfg = this.layers[i];
      const layerInput = this.lastActivations[this.lastActivations.length - 1];

      if (this.debug) {
        console.log(
          ` Processing L${i} (${cfg.type}). Input type=${layerInput.constructor.name}, len=${layerInput.length}`
        );
      }

      this.validateLayerInput(layerInput, cfg, i);

      const output = this.processLayer(layerInput, cfg, i);
      this.lastActivations.push(output);
    }

    return this.lastActivations[this.lastActivations.length - 1];
  }

  validateLayerInput(layerInput, cfg, layerIndex) {
    if (!(layerInput instanceof Float32Array)) {
      throw new Error(
        `L${layerIndex}(${cfg.type}): Internal error - input is not Float32Array.`
      );
    }
    if (layerInput.length !== cfg.inputSize) {
      throw new Error(
        `L${layerIndex}(${cfg.type}): Size mismatch. Expected ${cfg.inputSize}, got ${layerInput.length}.`
      );
    }
  }

  processLayer(layerInput, cfg, layerIndex) {
    switch (cfg.type) {
      case 'dense':
        return this.processDenseLayer(layerInput, cfg, layerIndex);
      case 'layernorm':
        return this.processLayerNorm(layerInput, cfg, layerIndex);
      case 'attention':
        return this.processAttentionLayer(layerInput, cfg, layerIndex);
      case 'dropout':
        return this.processDropoutLayer(layerInput, cfg, layerIndex);
      case 'softmax':
        return this.processSoftmaxLayer(layerInput, cfg, layerIndex);
      default:
        throw new Error(`Unknown layer type: ${cfg.type}`);
    }
  }

  processDenseLayer(layerInput, cfg, layerIndex) {
    const weights = this.weights[layerIndex];
    const biases = this.biases[layerIndex];
    
    this.validateDenseLayerParameters(weights, biases, layerIndex);
    
    const output = new Float32Array(cfg.outputSize);
    
    if (this.debug) {
      console.log(
        ` L${layerIndex} Dense: InputLen=${layerInput.length}, WeightLen=${weights.length}, BiasLen=${biases?.length}, OutputLen=${output.length}`
      );
    }

    for (let j = 0; j < cfg.outputSize; ++j) {
      let sum = biases ? biases[j] : 0;
      const weightRowOffset = j * cfg.inputSize;

      for (let k = 0; k < cfg.inputSize; ++k) {
        sum += layerInput[k] * weights[weightRowOffset + k];
      }

      output[j] = oblixActivations.apply(sum, cfg.activation);
    }

    return output;
  }

  validateDenseLayerParameters(weights, biases, layerIndex) {
    if (!weights) {
      throw new Error(`L${layerIndex} Dense: Weights not initialized.`);
    }
    if (!(weights instanceof Float32Array)) {
      throw new Error(
        `L${layerIndex} Dense: Weights internal error - not Float32Array.`
      );
    }
    if (biases && !(biases instanceof Float32Array)) {
      throw new Error(
        `L${layerIndex} Dense: Biases internal error - not Float32Array.`
      );
    }
  }

  processLayerNorm(layerInput, cfg, layerIndex) {
    const gammas = this.gammas[layerIndex];
    const betas = this.betas[layerIndex];
    
    const result = oblixLayerOps.layerNormForward(
      this,
      layerInput,
      gammas,
      betas
    );
    
    return result.output;
  }

  processAttentionLayer(layerInput, cfg, layerIndex) {
    return oblixLayerOps.attentionForward(
      this,
      layerInput,
      cfg.numHeads
    );
  }

  processDropoutLayer(layerInput, cfg, layerIndex) {
    return oblixLayerOps.dropoutForward(
      this,
      layerInput,
      cfg.rate
    );
  }

  processSoftmaxLayer(layerInput, cfg, layerIndex) {
    return oblixLayerOps.softmaxForward(this, layerInput);
  }

  /**
   *
   */
  save(name = 'model') {
    if (!this.layers || this.layers.length === 0) {
      console.warn('Save: Empty model.');
    }
    const numLayers = this.layers.length;

    const ensureLen = (arrName, dv = null) => {
      const currentArr = this[arrName];
      if (!Array.isArray(currentArr) || currentArr.length !== numLayers) {
        console.warn(`Adjusting length of ${arrName} to ${numLayers}`);
        const newArr = Array(numLayers).fill(dv);
        if (Array.isArray(currentArr)) {
          for (let i = 0; i < Math.min(numLayers, currentArr.length); ++i)
            newArr[i] = currentArr[i];
        }
        return newArr;
      }
      return currentArr;
    };

    const optimizerState = {
      t: this.t,
      m_dw: ensureLen('m_dw'),
      v_dw: ensureLen('v_dw'),
      m_db: ensureLen('m_db'),
      v_db: ensureLen('v_db'),
      m_dgamma: ensureLen('m_dgamma'),
      v_dgamma: ensureLen('v_dgamma'),
      m_dbeta: ensureLen('m_dbeta'),
      v_dbeta: ensureLen('v_dbeta'),
      s_dw: ensureLen('s_dw'),
      s_db: ensureLen('s_db'),
      s_dgamma: ensureLen('s_dgamma'),
      s_dbeta: ensureLen('s_dbeta')
    };

    const data = {
      weights: ensureLen('weights'),
      biases: ensureLen('biases'),
      gammas: ensureLen('gammas'),
      betas: ensureLen('betas'),
      layers: this.layers,
      details: this.details,
      usePositionalEncoding: this.usePositionalEncoding,
      optimizerState: optimizerState
    };

    try {
      if (this.debug) console.log('Preparing data object:', data);
      const jsonStr = JSON.stringify(data);
      if (this.debug) console.log('Stringified JSON (length):', jsonStr.length);

      if (this.debug && data.weights[0] instanceof Float32Array) {
        console.log(
          'Sample stringified weight (should be object):',
          jsonStr.substring(0, 500).includes('"weights":[{"0":')
        );
      }

      // Check if we're in a browser environment
      if (typeof document !== 'undefined' && typeof URL !== 'undefined') {
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${name}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        if (this.debug) console.log(`Model saved: ${name}.json`);
      } else {
        // Node.js environment - just log the data
        if (this.debug) console.log(`Model data prepared for save: ${name}.json`);
        console.log('Save functionality requires browser environment');
      }
    } catch (e) {
      console.error('Save failed.', e);
      if (this.debug) console.error(' Error during stringify or download.');
    }
  }

  /**
   *
   */
  load(callback) {
    // Check if we're in a browser environment
    if (typeof document === 'undefined') {
      console.log('Load functionality requires browser environment');
      if (callback) callback(new Error('Load functionality requires browser environment'));
      return;
    }
    
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.style.display = 'none';
    const handleListener = (event) => {
      const file = event.target.files[0];
      if (!file) {
        cleanup();
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        try {
          if (this.debug) console.log(' Reading file text...');
          const data = JSON.parse(text);
          if (!data.layers || !Array.isArray(data.layers))
            throw new Error('Invalid model: \'layers\' missing.');
          if (this.debug)
            console.log(' Parsed data, layers found:', data.layers.length);

          this.reset();
          this.layers = data.layers;
          this.details = data.details || {};
          this.usePositionalEncoding = data.usePositionalEncoding || false;
          const numLayers = this.layers.length;

          const loadAndReconstruct = (arrName, sourceObj, expectedLen) => {
            let loadedArr = sourceObj?.[arrName] || [];
            if (!Array.isArray(loadedArr)) {
              console.warn(
                ` ${arrName} in loaded data is not an array, creating default.`
              );
              loadedArr = [];
            }

            if (loadedArr.length !== expectedLen) {
              console.warn(
                ` ${arrName} length mismatch (expected ${expectedLen}, got ${loadedArr.length}). Adjusting...`
              );
              const adjustedArr = Array(expectedLen).fill(null);
              for (let i = 0; i < Math.min(expectedLen, loadedArr.length); ++i)
                adjustedArr[i] = loadedArr[i];
              loadedArr = adjustedArr;
            }

            return loadedArr.map((item, index) => {
              if (
                item !== null &&
                typeof item === 'object' &&
                item.hasOwnProperty('0')
              ) {
                const values = Object.values(item);

                const allNumbers = values.every(
                  (v) => typeof v === 'number' && isFinite(v)
                );

                if (allNumbers) {
                  const reconstructed = new Float32Array(values);
                  if (this.debug)
                    console.log(
                      ` Reconstructed Float32Array for ${arrName}[${index}], Length: ${reconstructed.length}`
                    );
                  return reconstructed;
                } else {
                  console.warn(
                    ` Object for ${arrName}[${index}] looks like Float32Array but check failed. Logging values:`
                  );

                  let loggedCount = 0;
                  for (let k = 0; k < values.length && loggedCount < 5; k++) {
                    const v = values[k];
                    if (typeof v !== 'number' || !isFinite(v)) {
                      console.warn(
                        `  - ${arrName}[${index}], Value[${k}]: Type=${typeof v}, Value=${v}`
                      );
                      loggedCount++;
                    }
                  }
                  if (loggedCount === 0 && values.length > 0) {
                    console.warn(
                      `  - ${arrName}[${index}]: Check failed but couldn't find non-numeric/non-finite value? First value:`,
                      values[0]
                    );
                  }

                  return null;
                }
              } else if (item instanceof Float32Array) {
                if (this.debug)
                  console.log(
                    ` Item ${arrName}[${index}] is already Float32Array? Length: ${item.length}`
                  );
                return item;
              } else if (item === null) {
                return null;
              } else {
                console.warn(
                  ` Unexpected item type for ${arrName}[${index}] (Type: ${typeof item}). Setting to null. Value:`,
                  item
                );
                return null;
              }
            });
          };

          this.weights = loadAndReconstruct('weights', data, numLayers);
          this.biases = loadAndReconstruct('biases', data, numLayers);
          this.gammas = loadAndReconstruct('gammas', data, numLayers);
          this.betas = loadAndReconstruct('betas', data, numLayers);
          this.masks = Array(numLayers).fill(null);

          const optState = data.optimizerState || {};
          this.t = optState.t || 0;
          if (this.debug) console.log(' Loading optimizer state...');
          this.m_dw = loadAndReconstruct('m_dw', optState, numLayers);
          this.v_dw = loadAndReconstruct('v_dw', optState, numLayers);
          this.m_db = loadAndReconstruct('m_db', optState, numLayers);
          this.v_db = loadAndReconstruct('v_db', optState, numLayers);
          this.m_dgamma = loadAndReconstruct('m_dgamma', optState, numLayers);
          this.v_dgamma = loadAndReconstruct('v_dgamma', optState, numLayers);
          this.m_dbeta = loadAndReconstruct('m_dbeta', optState, numLayers);
          this.v_dbeta = loadAndReconstruct('v_dbeta', optState, numLayers);
          this.s_dw = loadAndReconstruct('s_dw', optState, numLayers);
          this.s_db = loadAndReconstruct('s_db', optState, numLayers);
          this.s_dgamma = loadAndReconstruct('s_dgamma', optState, numLayers);
          this.s_dbeta = loadAndReconstruct('s_dbeta', optState, numLayers);

          this.lastActivations = null;
          this.forwardCache = null;
          this.isTraining = false;
          if (callback) callback();
          if (this.debug)
            console.log(
              ' Model loaded successfully. Stored parameters/states should be Float32Arrays or null.'
            );
          if (this.debug && this.weights.length > 0)
            console.log(
              ' Sample loaded weight type:',
              this.weights[0] instanceof Float32Array
                ? 'Float32Array'
                : typeof this.weights[0]
            );
        } catch (err) {
          console.error('Load failed:', err);
          alert(`Error loading model: ${err.message}`);
          if (this.debug)
            console.error(' Error during parsing or reconstruction.');
          if (callback) callback(err);
        } finally {
          cleanup();
        }
      };
      reader.onerror = (err) => {
        console.error('File read error:', err);
        alert('Error reading file.');
        cleanup();
        if (callback) callback(err);
      };
      reader.readAsText(file);
    };
    const cleanup = () => {
      input.removeEventListener('change', handleListener);
      document.body.removeChild(input);
    };
    input.addEventListener('change', handleListener);
    document.body.appendChild(input);
    input.click();
  }

  /**
   * Validates input for forward pass.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {*} input - Input to validate
   * @returns {Float32Array|null} Validated input or null if invalid
   */
  validateForwardInput(input) {
    if (!Array.isArray(input)) {
      console.warn('Skip invalid data');
      return null;
    }
    return input;
  }

  /**
   * Applies positional encoding to input if enabled.
   * Business rule: Positional encoding helps with sequence processing.
   *
   * @param {Float32Array} input - Input to encode
   * @returns {Float32Array} Encoded input
   */
  applyPositionalEncodingToInput(input) {
    if (this.usePositionalEncoding) {
      return oblixUtils.positionalEncoding(input);
    }
    return input;
  }

  /**
   * Initializes forward pass cache.
   * Business rule: Caching improves performance and enables backpropagation.
   *
   * @param {Float32Array} initialActivation - Initial activation
   * @returns {Object} Forward cache object
   */
  initializeForwardCache(initialActivation) {
    return {
      activations: [initialActivation],
      rawValues: [],
      layerNormIntermediates: [],
      attentionIntermediates: [],
      softmaxOutputs: []
    };
  }

  /**
   * Validates layer input for forward pass.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} layerInput - Layer input to validate
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @throws {Error} If input is invalid
   */
  validateLayerInputForForward(layerInput, layerConfig, layerIndex) {
    if (!(layerInput instanceof Float32Array)) {
      throw new Error(
        `L${layerIndex}(${layerConfig.type}): Internal error - input is not Float32Array.`
      );
    }
    if (layerInput.length !== layerConfig.inputSize) {
      throw new Error(
        `L${layerIndex}(${layerConfig.type}): Sz mismatch ${layerInput.length}!=${layerConfig.inputSize}`
      );
    }
  }

  /**
   * Processes dense layer forward pass.
   * Business rule: Dense layers perform linear transformations.
   *
   * @param {Float32Array} layerInput - Layer input
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @returns {Float32Array} Layer output
   * @throws {Error} If layer processing fails
   */
  processDenseLayerForward(layerInput, layerConfig, layerIndex) {
    const weights = this.weights[layerIndex];
    const biases = this.biases[layerIndex];
    
    if (!(weights instanceof Float32Array)) {
      throw new Error(`L${layerIndex} Dense: Weights not Float32Array.`);
    }
    if (biases && !(biases instanceof Float32Array)) {
      throw new Error(`L${layerIndex} Dense: Bias not Float32Array.`);
    }

    const rawSums = new Float32Array(layerConfig.outputSize);
    for (let j = 0; j < layerConfig.outputSize; ++j) {
      let sum = biases ? biases[j] : 0;
      const weightRowOffset = j * layerConfig.inputSize;
      for (let k = 0; k < layerConfig.inputSize; ++k) {
        sum += layerInput[k] * weights[weightRowOffset + k];
      }
      rawSums[j] = sum;
    }
    this.forwardCache.rawValues[layerIndex] = rawSums;

    const output = new Float32Array(layerConfig.outputSize);
    for (let j = 0; j < layerConfig.outputSize; ++j) {
      output[j] = oblixActivations.apply(rawSums[j], layerConfig.activation);
    }

    return output;
  }

  /**
   * Processes layer forward pass based on type.
   * Business rule: Each layer type has specific processing logic.
   *
   * @param {Float32Array} layerInput - Layer input
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @returns {Float32Array} Layer output
   * @throws {Error} If layer type is unknown
   */
  processLayerForward(layerInput, layerConfig, layerIndex) {
    this.forwardCache.rawValues[layerIndex] = null;

    switch (layerConfig.type) {
    case 'dense':
      return this.processDenseLayerForward(layerInput, layerConfig, layerIndex);
    case 'layernorm':
      return oblixLayerOps.layerNormForward(
        this,
        layerInput,
        this.gammas[layerIndex],
        this.betas[layerIndex]
      ).output;
    case 'attention':
      return oblixLayerOps.attentionForward(
        this,
        layerInput,
        layerConfig.numHeads
      );
    case 'dropout':
      return oblixLayerOps.dropoutForward(
        this,
        layerInput,
        layerConfig.rate
      );
    case 'softmax':
      return oblixLayerOps.softmaxForward(this, layerInput);
    default:
      throw new Error(`Fwd Pass: Unknown type ${layerConfig.type}`);
    }
  }

  /**
   * Validates layer output for forward pass.
   * Business rule: Output validation ensures data integrity.
   *
   * @param {Float32Array} output - Layer output to validate
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @throws {Error} If output is invalid
   */
  validateLayerOutputForForward(output, layerConfig, layerIndex) {
    if (!(output instanceof Float32Array)) {
      throw new Error(
        `L${layerIndex}(${layerConfig.type}): Internal error - output is not Float32Array.`
      );
    }
  }

  /**
   * Performs forward pass through the neural network.
   * Business rule: Forward pass computes predictions from input.
   *
   * @param {*} input - Network input
   * @returns {Float32Array|null} Network output or null if invalid
   */
  forward(input) {
    const validatedInput = this.validateForwardInput(input);
    if (!validatedInput) return null;

    const encodedInput = this.applyPositionalEncodingToInput(validatedInput);
    const initialActivation = encodedInput instanceof Float32Array
      ? encodedInput
      : new Float32Array(encodedInput);
    
    this.forwardCache = this.initializeForwardCache(initialActivation);
    let layerInput = this.forwardCache.activations[0];

    for (let i = 0; i < this.layers.length; i++) {
      const layerConfig = this.layers[i];

      try {
        this.validateLayerInputForForward(layerInput, layerConfig, i);
        const output = this.processLayerForward(layerInput, layerConfig, i);
        this.validateLayerOutputForForward(output, layerConfig, i);
        
        this.forwardCache.activations.push(output);
        layerInput = output;
      } catch (error) {
        console.error(`Fwd L${i}(${layerConfig.type}) Err:`, error);
        this.isTraining = false;
        throw error;
      }
    }

    return layerInput;
  }

  /**
   * Validates backward pass inputs.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} targetOutput - Target output
   * @throws {Error} If inputs are invalid
   */
  validateBackwardInputs(finalOutput, targetOutput) {
    if (finalOutput.length !== targetOutput.length) {
      throw new Error('Output/Target len mismatch');
    }
  }

  /**
   * Creates one-hot target for cross-entropy loss.
   * Business rule: One-hot encoding is required for categorical cross-entropy.
   *
   * @param {Float32Array} targetOutput - Target output
   * @param {number} outputLength - Length of output
   * @returns {Float32Array} One-hot encoded target
   * @throws {Error} If target format is unclear
   */
  createOneHotTarget(targetOutput, outputLength) {
    const oneHotTarget = new Float32Array(outputLength).fill(0);
    
    if (
      targetOutput.length === 1 &&
      Number.isInteger(targetOutput[0]) &&
      targetOutput[0] >= 0 &&
      targetOutput[0] < outputLength
    ) {
      oneHotTarget[targetOutput[0]] = 1;
    } else if (targetOutput.length === outputLength) {
      for (let i = 0; i < targetOutput.length; ++i) {
        oneHotTarget[i] = targetOutput[i];
      }
    } else {
      throw new Error('CE target unclear');
    }
    
    return oneHotTarget;
  }

  /**
   * Calculates cross-entropy loss with softmax.
   * Business rule: Cross-entropy with softmax is standard for classification.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} oneHotTarget - One-hot target
   * @returns {Object} Loss and gradient
   */
  calculateCrossEntropyWithSoftmax(finalOutput, oneHotTarget) {
    const epsilon = 1e-9;
    let loss = 0;
    
    for (let i = 0; i < finalOutput.length; ++i) {
      loss -= oneHotTarget[i] * Math.log(finalOutput[i] + epsilon);
    }

    const gradient = new Float32Array(finalOutput.length);
    for (let i = 0; i < finalOutput.length; ++i) {
      gradient[i] = finalOutput[i] - oneHotTarget[i];
    }

    return { loss, gradient };
  }

  /**
   * Calculates binary cross-entropy loss.
   * Business rule: Binary cross-entropy is used for binary classification.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} targetOutput - Target output
   * @returns {Object} Loss and gradient
   * @throws {Error} If output/target dimensions are invalid
   */
  calculateBinaryCrossEntropy(finalOutput, targetOutput) {
    if (finalOutput.length !== 1 || targetOutput.length !== 1) {
      throw new Error('BCE needs single out/target');
    }
    
    const epsilon = 1e-9;
    const prediction = finalOutput[0];
    const target = targetOutput[0];
    
    const loss = -(
      target * Math.log(prediction + epsilon) +
      (1 - target) * Math.log(1 - prediction + epsilon)
    );
    
    const gradient = new Float32Array([prediction - target]);
    
    return { loss, gradient };
  }

  /**
   * Calculates simple cross-entropy loss.
   * Business rule: Fallback for cross-entropy without proper final layer.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} targetOutput - Target output
   * @returns {Object} Loss and gradient
   */
  calculateSimpleCrossEntropy(finalOutput, targetOutput) {
    console.warn('CE w/o final softmax/sigmoid, using simple diff');
    
    const gradient = new Float32Array(finalOutput.length);
    for (let i = 0; i < finalOutput.length; ++i) {
      gradient[i] = finalOutput[i] - targetOutput[i];
    }
    
    const loss = 0.5 * gradient.reduce((sum, error) => sum + error * error, 0);
    
    return { loss, gradient };
  }

  /**
   * Calculates cross-entropy loss based on final layer type.
   * Business rule: Loss calculation depends on final layer configuration.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} targetOutput - Target output
   * @param {Object} layerConfig - Final layer configuration
   * @returns {Object} Loss and gradient
   */
  calculateCrossEntropyLoss(finalOutput, targetOutput, layerConfig) {
    const hasSoftmax = layerConfig.type === 'softmax' ||
      (layerConfig.type === 'dense' && layerConfig.activation === 'softmax');
    const hasSigmoid = layerConfig.type === 'dense' && layerConfig.activation === 'sigmoid';

    if (hasSoftmax) {
      const oneHotTarget = this.createOneHotTarget(targetOutput, finalOutput.length);
      return this.calculateCrossEntropyWithSoftmax(finalOutput, oneHotTarget);
    } else if (hasSigmoid) {
      return this.calculateBinaryCrossEntropy(finalOutput, targetOutput);
    } else {
      return this.calculateSimpleCrossEntropy(finalOutput, targetOutput);
    }
  }

  /**
   * Calculates mean squared error loss.
   * Business rule: MSE is standard for regression tasks.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} targetOutput - Target output
   * @returns {Object} Loss and gradient
   */
  calculateMeanSquaredErrorLoss(finalOutput, targetOutput) {
    const gradient = new Float32Array(finalOutput.length);
    let loss = 0;
    
    for (let i = 0; i < finalOutput.length; ++i) {
      const diff = finalOutput[i] - targetOutput[i];
      gradient[i] = diff;
      loss += diff * diff;
    }
    
    loss *= 0.5;
    
    return { loss, gradient };
  }

  /**
   * Calculates initial gradient based on loss function.
   * Business rule: Initial gradient depends on loss function choice.
   *
   * @param {Float32Array} finalOutput - Final network output
   * @param {Float32Array} targetOutput - Target output
   * @param {Object} config - Training configuration
   * @returns {Float32Array} Initial gradient
   */
  calculateInitialGradient(finalOutput, targetOutput, config) {
    if (config.lossFunction === 'crossentropy') {
      const lastLayer = this.layers[this.layers.length - 1];
      const { gradient } = this.calculateCrossEntropyLoss(finalOutput, targetOutput, lastLayer);
      return gradient;
    } else {
      const { gradient } = this.calculateMeanSquaredErrorLoss(finalOutput, targetOutput);
      return gradient;
    }
  }

  /**
   * Validates gradient for backward pass.
   * Business rule: Gradient validation prevents propagation errors.
   *
   * @param {Float32Array} gradient - Gradient to validate
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @returns {Float32Array} Validated gradient or zeros if invalid
   */
  validateBackwardGradient(gradient, layerConfig, layerIndex) {
    if (
      !(gradient instanceof Float32Array) ||
      gradient.length !== layerConfig.outputSize
    ) {
      console.warn(
        `Bkwd L${layerIndex}(${layerConfig.type}): Invalid gradient. Type: ${gradient?.constructor?.name}, Len: ${gradient?.length}. Expected Len: ${layerConfig.outputSize}. Using zeros.`
      );
      return new Float32Array(layerConfig.outputSize).fill(0);
    }
    return gradient;
  }

  /**
   * Processes dense layer backward pass.
   * Business rule: Dense layer backpropagation computes weight gradients.
   *
   * @param {Float32Array} gradient - Input gradient
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @returns {Float32Array} Output gradient
   * @throws {Error} If layer processing fails
   */
  processDenseLayerBackward(gradient, layerConfig, layerIndex) {
    const weights = this.weights[layerIndex];
    const biases = this.biases[layerIndex];
    const rawValues = this.forwardCache.rawValues[layerIndex];
    const activation = layerConfig.activation;
    const inputSize = layerConfig.inputSize;
    const outputSize = layerConfig.outputSize;
    const previousActivation = this.forwardCache.activations[layerIndex];

    if (!(rawValues instanceof Float32Array)) {
      throw new Error(`L${layerIndex} Dense Bkwd: Missing or invalid raw values cache.`);
    }
    if (!(weights instanceof Float32Array)) {
      throw new Error(`L${layerIndex} Dense Bkwd: Weights not Float32Array.`);
    }
    if (!(previousActivation instanceof Float32Array)) {
      throw new Error(`L${layerIndex} Dense Bkwd: Previous activation not Float32Array.`);
    }

    const delta = new Float32Array(outputSize);
    for (let j = 0; j < outputSize; ++j) {
      const derivative = oblixActivations.derivative(rawValues[j], activation);
      if (typeof derivative !== 'number' || !isFinite(derivative)) {
        console.warn(
          `L${layerIndex} Dense, j=${j}: Deriv NaN/Inf. Activation: ${activation}, Raw Input: ${rawValues[j]}, Derivative: ${derivative}`
        );
        delta[j] = 0;
      } else {
        delta[j] = gradient[j] * derivative;
      }
    }

    const inputGradient = new Float32Array(inputSize).fill(0);
    for (let k = 0; k < inputSize; k++) {
      for (let j = 0; j < outputSize; j++) {
        const weightIndex = j * inputSize + k;
        inputGradient[k] += delta[j] * weights[weightIndex];
      }
    }

    return inputGradient;
  }

  /**
   * Processes layer backward pass based on type.
   * Business rule: Each layer type has specific backpropagation logic.
   *
   * @param {Float32Array} gradient - Input gradient
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @returns {Float32Array} Output gradient
   * @throws {Error} If layer type is unknown
   */
  processLayerBackward(gradient, layerConfig, layerIndex) {
    switch (layerConfig.type) {
    case 'dense':
      return this.processDenseLayerBackward(gradient, layerConfig, layerIndex);
    case 'layernorm':
      const layerNormCache = this.forwardCache.layerNormIntermediates[layerIndex];
      if (!layerNormCache) throw new Error(`L${layerIndex} LN Bkwd: Missing cache`);
      const { dInput } = oblixLayerOps.layerNormBackward(this, gradient, layerNormCache);
      return dInput;
    case 'attention':
      const attentionCache = this.forwardCache.attentionIntermediates[layerIndex];
      if (!attentionCache) throw new Error(`L${layerIndex} Attn Bkwd: Missing cache`);
      const { dInput: attentionDInput } = oblixLayerOps.attentionBackward(
        this,
        gradient,
        attentionCache
      );
      return attentionDInput;
    case 'dropout':
      return oblixLayerOps.dropoutBackward(this, gradient, layerIndex);
    case 'softmax':
      return oblixLayerOps.softmaxBackward(this, gradient, layerIndex);
    default:
      throw new Error(`Bkwd Pass: Unknown type ${layerConfig.type}`);
    }
  }

  /**
   * Validates layer output gradient for backward pass.
   * Business rule: Output gradient validation ensures data integrity.
   *
   * @param {Float32Array} outputGradient - Layer output gradient to validate
   * @param {Object} layerConfig - Layer configuration
   * @param {number} layerIndex - Layer index
   * @throws {Error} If output gradient is invalid
   */
  validateLayerOutputGradient(outputGradient, layerConfig, layerIndex) {
    if (!(outputGradient instanceof Float32Array)) {
      throw new Error(
        `Bkwd L${layerIndex}(${layerConfig.type}): Internal error - output gradient is not Float32Array.`
      );
    }
  }

  /**
   * Performs backward pass through the neural network.
   * Business rule: Backward pass computes gradients for parameter updates.
   *
   * @param {Float32Array} target - Target output
   * @param {Object} config - Training configuration
   */
  backward(target, config) {
    const finalOutput = this.forwardCache.activations[this.forwardCache.activations.length - 1];
    const targetOutput = target;
    
    this.validateBackwardInputs(finalOutput, targetOutput);
    
    const initialGradient = this.calculateInitialGradient(finalOutput, targetOutput, config);
    let currentGradient = initialGradient;

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layerConfig = this.layers[i];
      const previousActivation = this.forwardCache.activations[i];

      const validatedGradient = this.validateBackwardGradient(currentGradient, layerConfig, i);
      if (validatedGradient.every(val => val === 0)) {
        currentGradient = new Float32Array(layerConfig.outputSize).fill(0);
        continue;
      }

      try {
        const outputGradient = this.processLayerBackward(validatedGradient, layerConfig, i);
        this.validateLayerOutputGradient(outputGradient, layerConfig, i);
        currentGradient = outputGradient;
      } catch (error) {
        console.error(`Bkwd L${i}(${layerConfig.type}) Err:`, error);
        this.isTraining = false;
        throw error;
      }
    }
  }

  accumulateGradients(gradients) {
    // This method would accumulate gradients from the backward pass
    // For now, we'll implement a simplified version
    for (let i = 0; i < this.layers.length; i++) {
      const cfg = this.layers[i];
      
      if (cfg.type === 'dense') {
        const act_prev = this.forwardCache.activations[i];
        const raw = this.forwardCache.rawValues[i];
        
        if (gradients.weights[i]) {
          const delta = new Float32Array(cfg.outputSize);
          for (let j = 0; j < cfg.outputSize; ++j) {
            const deriv = oblixActivations.derivative(raw[j], cfg.activation);
            delta[j] = deriv;
          }
          
          for (let j = 0; j < cfg.outputSize; j++) {
            const weightRowOffset = j * cfg.inputSize;
            for (let k = 0; k < cfg.inputSize; k++) {
              gradients.weights[i][weightRowOffset + k] += delta[j] * act_prev[k];
            }
          }
        }
        
        if (gradients.biases[i]) {
          const delta = new Float32Array(cfg.outputSize);
          for (let j = 0; j < cfg.outputSize; ++j) {
            const deriv = oblixActivations.derivative(raw[j], cfg.activation);
            delta[j] = deriv;
          }
          
          for (let j = 0; j < cfg.outputSize; j++) {
            gradients.biases[i][j] += delta[j];
          }
        }
      }
    }
  }

  updateParameters(gradients, learningRate, config) {
    const updateOptions = {
      learningRate: learningRate,
      initialLearningRate: config.initialLearningRate,
      optimizer: config.optimizer,
      batchSize: 1, // This will be updated by the caller
      l2Lambda: config.l2Lambda,
      gradientClipValue: config.gradientClipValue,
      decayRate: this.decayRate
    };
    
    oblixOptimizers.updateParameters(
      this,
      gradients.weights,
      gradients.biases,
      gradients.gammas,
      gradients.betas,
      updateOptions
    );
  }
}

export { Oblix, oblixActivations, oblixLayerOps, oblixOptimizers, oblixUtils };
