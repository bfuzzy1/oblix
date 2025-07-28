export const oblixOptimizers = {
  /**
   * Initializes optimizer state arrays for all layers.
   * Business rule: State arrays must be properly initialized for training.
   *
   * @param {Object} context - Optimizer context
   * @param {number} numLayers - Number of layers
   */
  initializeStateArrays(context, numLayers) {
    context.t = 0;
    context.m_dw = Array(numLayers).fill(null);
    context.v_dw = Array(numLayers).fill(null);
    context.m_db = Array(numLayers).fill(null);
    context.v_db = Array(numLayers).fill(null);
    context.m_dgamma = Array(numLayers).fill(null);
    context.v_dgamma = Array(numLayers).fill(null);
    context.m_dbeta = Array(numLayers).fill(null);
    context.v_dbeta = Array(numLayers).fill(null);
    context.s_dw = Array(numLayers).fill(null);
    context.s_db = Array(numLayers).fill(null);
    context.s_dgamma = Array(numLayers).fill(null);
    context.s_dbeta = Array(numLayers).fill(null);
  },

  /**
   * Validates layer configuration for optimizer state initialization.
   * Business rule: Layer validation prevents runtime errors.
   *
   * @param {Object} cfg - Layer configuration
   * @param {Float32Array} weights - Weight array
   * @param {Float32Array} biases - Bias array
   * @param {Float32Array} gammas - Gamma array
   * @param {Float32Array} betas - Beta array
   * @returns {Object} Validation results
   */
  validateLayerForOptimizer(cfg, weights, biases, gammas, betas) {
    const reqWeights = cfg.type === 'dense' && weights instanceof Float32Array && weights.length > 0;
    const reqBiases = cfg.type === 'dense' && cfg.useBias && biases instanceof Float32Array && biases.length > 0;
    const reqLayerNorm = cfg.type === 'layernorm' && 
      gammas instanceof Float32Array && 
      betas instanceof Float32Array && 
      gammas.length === betas.length && 
      gammas.length > 0;

    return { reqWeights, reqBiases, reqLayerNorm };
  },

  /**
   * Initializes layer normalization optimizer state.
   * Business rule: Layer normalization requires separate gamma/beta state arrays.
   *
   * @param {Object} context - Optimizer context
   * @param {number} layerIndex - Layer index
   * @param {Float32Array} gammas - Gamma array
   * @param {Float32Array} betas - Beta array
   * @param {string} optimizer - Optimizer type
   */
  initializeLayerNormState(context, layerIndex, gammas, betas, optimizer) {
    const size = (gammas && gammas.length) || (betas && betas.length) || 0;
    
    if (size > 0) {
      if (optimizer === 'adam' || optimizer === 'adamw') {
        context.m_dgamma[layerIndex] = new Float32Array(size).fill(0);
        context.v_dgamma[layerIndex] = new Float32Array(size).fill(0);
        context.m_dbeta[layerIndex] = new Float32Array(size).fill(0);
        context.v_dbeta[layerIndex] = new Float32Array(size).fill(0);
      }
      if (optimizer === 'rmsprop') {
        context.s_dgamma[layerIndex] = new Float32Array(size).fill(0);
        context.s_dbeta[layerIndex] = new Float32Array(size).fill(0);
      }
    } else {
      console.error(`InitOpt L${layerIndex} LN err: gamma/beta missing or size 0`);
      context.m_dgamma[layerIndex] = null;
      context.v_dgamma[layerIndex] = null;
      context.m_dbeta[layerIndex] = null;
      context.v_dbeta[layerIndex] = null;
      context.s_dgamma[layerIndex] = null;
      context.s_dbeta[layerIndex] = null;
    }
  },

  /**
   * Initializes weight optimizer state for a layer.
   * Business rule: Weight state arrays must match weight array dimensions.
   *
   * @param {Object} context - Optimizer context
   * @param {number} layerIndex - Layer index
   * @param {Float32Array} weights - Weight array
   * @param {string} optimizer - Optimizer type
   */
  initializeWeightState(context, layerIndex, weights, optimizer) {
    if (optimizer === 'adam' || optimizer === 'rmsprop' || optimizer === 'adamw') {
      try {
        const size = weights.length;
        if (optimizer === 'adam' || optimizer === 'adamw') {
          context.m_dw[layerIndex] = new Float32Array(size).fill(0);
          context.v_dw[layerIndex] = new Float32Array(size).fill(0);
        }
        if (optimizer === 'rmsprop') {
          context.s_dw[layerIndex] = new Float32Array(size).fill(0);
        }
      } catch (error) {
        console.error(`InitOpt L${layerIndex} W err: ${error.message}`);
        context.m_dw[layerIndex] = null;
        context.v_dw[layerIndex] = null;
        context.s_dw[layerIndex] = null;
      }
    }
  },

  /**
   * Initializes bias optimizer state for a layer.
   * Business rule: Bias state arrays must match bias array dimensions.
   *
   * @param {Object} context - Optimizer context
   * @param {number} layerIndex - Layer index
   * @param {Float32Array} biases - Bias array
   * @param {string} optimizer - Optimizer type
   */
  initializeBiasState(context, layerIndex, biases, optimizer) {
    if (optimizer === 'adam' || optimizer === 'rmsprop' || optimizer === 'adamw') {
      try {
        const size = biases.length;
        if (optimizer === 'adam' || optimizer === 'adamw') {
          context.m_db[layerIndex] = new Float32Array(size).fill(0);
          context.v_db[layerIndex] = new Float32Array(size).fill(0);
        }
        if (optimizer === 'rmsprop') {
          context.s_db[layerIndex] = new Float32Array(size).fill(0);
        }
      } catch (error) {
        console.error(`InitOpt L${layerIndex} B err: ${error.message}`);
        context.m_db[layerIndex] = null;
        context.v_db[layerIndex] = null;
        context.s_db[layerIndex] = null;
      }
    }
  },

  /**
   * Initializes optimizer state for all layers.
   * Business rule: Optimizer state must be properly initialized for training.
   *
   * @param {Object} context - Optimizer context
   * @param {string} optimizer - Optimizer type
   */
  initializeState: function (context, optimizer) {
    const numLayers = context.layers.length;
    if (context.debug) {
      console.log(`Init optimizer state: ${optimizer}, ${numLayers} layers.`);
    }

    this.initializeStateArrays(context, numLayers);

    for (let i = 0; i < numLayers; i++) {
      const cfg = context.layers[i];
      if (!cfg) continue;

      const weights = context.weights[i];
      const biases = context.biases[i];
      const gammas = context.gammas[i];
      const betas = context.betas[i];

      const { reqWeights, reqBiases, reqLayerNorm } = this.validateLayerForOptimizer(cfg, weights, biases, gammas, betas);

      // Initialize layer normalization state
      if (cfg.type === 'layernorm') {
        this.initializeLayerNormState(context, i, gammas, betas, optimizer);
      }

      // Initialize weight state
      if (reqWeights) {
        this.initializeWeightState(context, i, weights, optimizer);
      }

      // Initialize bias state
      if (reqBiases) {
        this.initializeBiasState(context, i, biases, optimizer);
      }
    }

    if (context.debug) {
      console.log('Optimizer state init finished.');
    }
  },

  /**
   * Validates optimizer parameters.
   * Business rule: Parameter validation prevents runtime errors.
   *
   * @param {Object} options - Optimizer options
   * @returns {Object} Validated options
   */
  validateOptimizerOptions(options) {
    const {
      learningRate,
      initialLearningRate,
      optimizer,
      batchSize,
      l2Lambda,
      gradientClipValue,
      decayRate
    } = options;

    return {
      learningRate: learningRate || 0.001,
      initialLearningRate: initialLearningRate || learningRate || 0.001,
      optimizer: optimizer || 'sgd',
      batchSize: batchSize || 1,
      l2Lambda: l2Lambda || 0,
      gradientClipValue: gradientClipValue || 0,
      decayRate: decayRate || 0.9
    };
  },

  /**
   * Calculates effective learning rate for Adam optimizer.
   * Business rule: Adam requires bias correction for proper convergence.
   *
   * @param {number} learningRate - Current learning rate
   * @param {number} initialLearningRate - Initial learning rate
   * @param {Object} context - Optimizer context
   * @returns {number} Effective learning rate
   */
  calculateAdamLearningRate(learningRate, initialLearningRate, context) {
    const adamCorrectedBaseLR =
      (initialLearningRate * Math.sqrt(1 - context.beta2 ** context.t)) /
      (1 - context.beta1 ** context.t);
    return adamCorrectedBaseLR * (learningRate / initialLearningRate);
  },

  /**
   * Applies gradient clipping to prevent exploding gradients.
   * Business rule: Gradient clipping stabilizes training.
   *
   * @param {number} grad - Gradient value
   * @param {number} batchMult - Batch multiplier
   * @param {number} gradientClipValue - Maximum gradient value
   * @returns {number} Clipped gradient
   */
  applyGradientClipping(grad, batchMult, gradientClipValue) {
    const effectiveGrad = grad * batchMult;
    if (gradientClipValue <= 0) {
      return effectiveGrad;
    }
    return Math.max(
      -gradientClipValue,
      Math.min(gradientClipValue, effectiveGrad)
    );
  },

  /**
   * Updates parameter using Adam optimizer.
   * Business rule: Adam combines momentum and adaptive learning rates.
   *
   * @param {number} param - Parameter value
   * @param {number} clippedGrad - Clipped gradient
   * @param {number} mState - Momentum state
   * @param {number} vState - Velocity state
   * @param {Object} context - Optimizer context
   * @param {number} adamStepLR - Adam learning rate
   * @param {number} l2Lambda - L2 regularization
   * @param {string} optimizer - Optimizer type
   * @returns {Object} Updated parameter and states
   */
  updateParameterWithAdam(param, clippedGrad, mState, vState, context, adamStepLR, l2Lambda, optimizer) {
    let m = typeof mState === 'number' && isFinite(mState) ? mState : 0;
    let v = typeof vState === 'number' && isFinite(vState) ? vState : 0;

    m = context.beta1 * m + (1 - context.beta1) * clippedGrad;
    v = context.beta2 * v + (1 - context.beta2) * clippedGrad ** 2;
    
    const m_hat = m / (1 - context.beta1 ** context.t);
    const v_hat = v / (1 - context.beta2 ** context.t);

    const sqrt_v_hat = Math.sqrt(v_hat);
    let update = 0;
    
    if (
      isFinite(m_hat) &&
      isFinite(sqrt_v_hat) &&
      sqrt_v_hat + context.epsilon !== 0
    ) {
      update = (adamStepLR * m_hat) / (sqrt_v_hat + context.epsilon);
    }

    if (optimizer === 'adam' && l2Lambda > 0) {
      update += adamStepLR * l2Lambda * param;
    }

    return {
      newParam: param - update,
      newM: m,
      newV: v
    };
  },

  /**
   * Updates parameter using RMSprop optimizer.
   * Business rule: RMSprop adapts learning rates based on gradient magnitude.
   *
   * @param {number} param - Parameter value
   * @param {number} clippedGrad - Clipped gradient
   * @param {number} sState - RMS state
   * @param {number} stepLR - Learning rate
   * @param {number} decayRate - Decay rate
   * @param {Object} context - Optimizer context
   * @returns {Object} Updated parameter and state
   */
  updateParameterWithRMSprop(param, clippedGrad, sState, stepLR, decayRate, context) {
    let s = typeof sState === 'number' && isFinite(sState) ? sState : 0;
    
    s = decayRate * s + (1 - decayRate) * clippedGrad ** 2;
    const sqrt_s = Math.sqrt(s);
    
    let update = 0;
    if (
      isFinite(clippedGrad) &&
      isFinite(sqrt_s) &&
      sqrt_s + context.epsilon !== 0
    ) {
      update = (stepLR * clippedGrad) / (sqrt_s + context.epsilon);
    }

    return {
      newParam: param - update,
      newS: s
    };
  },

  /**
   * Updates parameter using SGD optimizer.
   * Business rule: SGD applies simple gradient descent with optional momentum.
   *
   * @param {number} param - Parameter value
   * @param {number} clippedGrad - Clipped gradient
   * @param {number} mState - Momentum state
   * @param {number} stepLR - Learning rate
   * @param {number} l2Lambda - L2 regularization
   * @param {Object} context - Optimizer context
   * @returns {Object} Updated parameter and state
   */
  updateParameterWithSGD(param, clippedGrad, mState, stepLR, l2Lambda, context) {
    let m = typeof mState === 'number' && isFinite(mState) ? mState : 0;
    
    m = context.beta1 * m + (1 - context.beta1) * clippedGrad;
    let update = stepLR * m;

    if (l2Lambda > 0) {
      update += stepLR * l2Lambda * param;
    }

    return {
      newParam: param - update,
      newM: m
    };
  },

  /**
   * Updates neural network parameters using the specified optimizer.
   * Business rule: Parameter updates must be numerically stable and efficient.
   *
   * @param {Object} context - Optimizer context
   * @param {Array} gradsW - Weight gradients
   * @param {Array} gradsB - Bias gradients
   * @param {Array} gradsGamma - Gamma gradients
   * @param {Array} gradsBeta - Beta gradients
   * @param {Object} options - Optimizer options
   */
  updateParameters: function (
    context,
    gradsW,
    gradsB,
    gradsGamma,
    gradsBeta,
    options
  ) {
    const validatedOptions = this.validateOptimizerOptions(options);
    const {
      learningRate,
      initialLearningRate,
      optimizer,
      batchSize,
      l2Lambda,
      gradientClipValue,
      decayRate
    } = validatedOptions;

    const batchMult = batchSize > 0 ? 1.0 / batchSize : 1.0;
    context.t++;

    if (context.debug) {
      console.log(
        ` T=${context.t}, LR=${learningRate.toExponential(3)}, BatchSize=${batchSize}, Opt=${optimizer}, L2=${l2Lambda}`
      );
    }

    const adamStepLR = this.calculateAdamLearningRate(learningRate, initialLearningRate, context);

    for (let i = 0; i < context.layers.length; i++) {
      const cfg = context.layers[i];
      const isDense = cfg.type === 'dense';
      const isLN = cfg.type === 'layernorm';

      const stepLR = learningRate;

      // Update weights for dense layers
      if (isDense && gradsW[i]) {
        for (let j = 0; j < gradsW[i].length; j++) {
          const param = context.weights[i][j];
          const grad = gradsW[i][j];
          const mState = context.m_dw[i]?.[j];
          const vState = context.v_dw[i]?.[j];
          const sState = context.s_dw[i]?.[j];

          const clippedGrad = this.applyGradientClipping(grad, batchMult, gradientClipValue);

          if (!isFinite(clippedGrad)) {
            continue;
          }

          let result;
          if (optimizer === 'adam' || optimizer === 'adamw') {
            result = this.updateParameterWithAdam(param, clippedGrad, mState, vState, context, adamStepLR, l2Lambda, optimizer);
            context.weights[i][j] = result.newParam;
            if (!context.m_dw[i]) context.m_dw[i] = [];
            if (!context.v_dw[i]) context.v_dw[i] = [];
            context.m_dw[i][j] = result.newM;
            context.v_dw[i][j] = result.newV;
          } else if (optimizer === 'rmsprop') {
            result = this.updateParameterWithRMSprop(param, clippedGrad, sState, stepLR, decayRate, context);
            context.weights[i][j] = result.newParam;
            if (!context.s_dw[i]) context.s_dw[i] = [];
            context.s_dw[i][j] = result.newS;
          } else {
            result = this.updateParameterWithSGD(param, clippedGrad, mState, stepLR, l2Lambda, context);
            context.weights[i][j] = result.newParam;
            if (!context.m_dw[i]) context.m_dw[i] = [];
            context.m_dw[i][j] = result.newM;
          }
        }
      }

      // Update biases for dense layers
      if (isDense && gradsB[i]) {
        for (let j = 0; j < gradsB[i].length; j++) {
          const param = context.biases[i][j];
          const grad = gradsB[i][j];
          const mState = context.m_db[i]?.[j];
          const vState = context.v_db[i]?.[j];
          const sState = context.s_db[i]?.[j];

          const clippedGrad = this.applyGradientClipping(grad, batchMult, gradientClipValue);

          if (!isFinite(clippedGrad)) {
            continue;
          }

          let result;
          if (optimizer === 'adam' || optimizer === 'adamw') {
            result = this.updateParameterWithAdam(param, clippedGrad, mState, vState, context, adamStepLR, 0, optimizer);
            context.biases[i][j] = result.newParam;
            if (!context.m_db[i]) context.m_db[i] = [];
            if (!context.v_db[i]) context.v_db[i] = [];
            context.m_db[i][j] = result.newM;
            context.v_db[i][j] = result.newV;
          } else if (optimizer === 'rmsprop') {
            result = this.updateParameterWithRMSprop(param, clippedGrad, sState, stepLR, decayRate, context);
            context.biases[i][j] = result.newParam;
            if (!context.s_db[i]) context.s_db[i] = [];
            context.s_db[i][j] = result.newS;
          } else {
            result = this.updateParameterWithSGD(param, clippedGrad, mState, stepLR, 0, context);
            context.biases[i][j] = result.newParam;
            if (!context.m_db[i]) context.m_db[i] = [];
            context.m_db[i][j] = result.newM;
          }
        }
      }

      // Update layer normalization parameters
      if (isLN) {
        if (gradsGamma[i]) {
          for (let j = 0; j < gradsGamma[i].length; j++) {
            const param = context.gammas[i][j];
            const grad = gradsGamma[i][j];
            const mState = context.m_dgamma[i]?.[j];
            const vState = context.v_dgamma[i]?.[j];
            const sState = context.s_dgamma[i]?.[j];

            const clippedGrad = this.applyGradientClipping(grad, batchMult, gradientClipValue);

            if (!isFinite(clippedGrad)) {
              continue;
            }

            let result;
            if (optimizer === 'adam' || optimizer === 'adamw') {
              result = this.updateParameterWithAdam(param, clippedGrad, mState, vState, context, adamStepLR, 0, optimizer);
              context.gammas[i][j] = result.newParam;
              if (!context.m_dgamma[i]) context.m_dgamma[i] = [];
              if (!context.v_dgamma[i]) context.v_dgamma[i] = [];
              context.m_dgamma[i][j] = result.newM;
              context.v_dgamma[i][j] = result.newV;
            } else if (optimizer === 'rmsprop') {
              result = this.updateParameterWithRMSprop(param, clippedGrad, sState, stepLR, decayRate, context);
              context.gammas[i][j] = result.newParam;
              if (!context.s_dgamma[i]) context.s_dgamma[i] = [];
              context.s_dgamma[i][j] = result.newS;
            } else {
              result = this.updateParameterWithSGD(param, clippedGrad, mState, stepLR, 0, context);
              context.gammas[i][j] = result.newParam;
              if (!context.m_dgamma[i]) context.m_dgamma[i] = [];
              context.m_dgamma[i][j] = result.newM;
            }
          }
        }

        if (gradsBeta[i]) {
          for (let j = 0; j < gradsBeta[i].length; j++) {
            const param = context.betas[i][j];
            const grad = gradsBeta[i][j];
            const mState = context.m_dbeta[i]?.[j];
            const vState = context.v_dbeta[i]?.[j];
            const sState = context.s_dbeta[i]?.[j];

            const clippedGrad = this.applyGradientClipping(grad, batchMult, gradientClipValue);

            if (!isFinite(clippedGrad)) {
              continue;
            }

            let result;
            if (optimizer === 'adam' || optimizer === 'adamw') {
              result = this.updateParameterWithAdam(param, clippedGrad, mState, vState, context, adamStepLR, 0, optimizer);
              context.betas[i][j] = result.newParam;
              if (!context.m_dbeta[i]) context.m_dbeta[i] = [];
              if (!context.v_dbeta[i]) context.v_dbeta[i] = [];
              context.m_dbeta[i][j] = result.newM;
              context.v_dbeta[i][j] = result.newV;
            } else if (optimizer === 'rmsprop') {
              result = this.updateParameterWithRMSprop(param, clippedGrad, sState, stepLR, decayRate, context);
              context.betas[i][j] = result.newParam;
              if (!context.s_dbeta[i]) context.s_dbeta[i] = [];
              context.s_dbeta[i][j] = result.newS;
            } else {
              result = this.updateParameterWithSGD(param, clippedGrad, mState, stepLR, 0, context);
              context.betas[i][j] = result.newParam;
              if (!context.m_dbeta[i]) context.m_dbeta[i] = [];
              context.m_dbeta[i][j] = result.newM;
            }
          }
        }
      }
    }
  }
};

