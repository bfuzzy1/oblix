import { oblixUtils } from './utils.js';

export const oblixLayerOps = {
  /**
   * Validates attention forward inputs.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} input - Input array
   * @param {number} numHeads - Number of attention heads
   * @returns {Object} Validation result with input and headSize or null
   */
  validateAttentionForwardInputs: function (input, numHeads) {
    if (!(input instanceof Float32Array)) {
      console.warn(' Input not Float32Array.', input);
      input = new Float32Array(input);
    }
    const inputDim = input.length;
    if (inputDim === 0) return { input: new Float32Array(0), headSize: 0 };
    if (
      numHeads <= 0 ||
      !Number.isInteger(numHeads) ||
      inputDim % numHeads !== 0
    ) {
      console.error(`Attn Error: Dim ${inputDim} not divisible by ${numHeads}`);
      return null;
    }
    return { input, headSize: inputDim / numHeads };
  },

  /**
   * Calculates attention scores for a single head.
   * Business rule: Attention scores are computed as dot products of queries and keys.
   *
   * @param {Float32Array} q_head - Query head
   * @param {Float32Array} k_head - Key head
   * @param {number} headSize - Size of each head
   * @returns {Array} Attention scores matrix
   */
  calculateAttentionScores: function (q_head, k_head, headSize) {
    return Array.from({ length: headSize }, (_, i) =>
      Array.from({ length: headSize }, (_, j) => q_head[i] * k_head[j])
    );
  },

  /**
   * Applies softmax to attention scores.
   * Business rule: Softmax normalizes attention weights to sum to 1.
   *
   * @param {Array} scaled - Scaled attention scores
   * @returns {Array} Attention weights after softmax
   */
  applyAttentionSoftmax: function (scaled) {
    return scaled.map((r) => {
      if (r.length === 0) return [];
      const max = Math.max(...r, -Infinity);
      const exps = r.map((s) => Math.exp(s - max));
      const sum = exps.reduce((a, b) => a + b, 1e-9);
      return exps.map((e) => e / sum);
    });
  },

  /**
   * Computes weighted sum for attention output.
   * Business rule: Attention output is the weighted sum of values.
   *
   * @param {Array} attn - Attention weights
   * @param {Float32Array} v_head - Value head
   * @param {number} headSize - Size of each head
   * @returns {Float32Array} Weighted sum output
   */
  computeAttentionWeightedSum: function (attn, v_head, headSize) {
    const output = new Float32Array(headSize);
    for (let i = 0; i < headSize; i++) {
      let weighted_sum = 0;
      for (let j = 0; j < headSize; j++) {
        weighted_sum += attn[i][j] * v_head[j];
      }
      output[i] = weighted_sum;
    }
    return output;
  },

  /**
   * Processes a single attention head.
   * Business rule: Each attention head operates independently on a subset of the input.
   *
   * @param {Float32Array} input - Input array
   * @param {number} headIndex - Head index
   * @param {number} headSize - Size of each head
   * @returns {Object} Head output and attention weights
   */
  processAttentionHead: function (input, headIndex, headSize) {
    const start = headIndex * headSize;
    const end = start + headSize;

    const q_head = input.slice(start, end);
    const k_head = q_head;
    const v_head = q_head;

    const scores = this.calculateAttentionScores(q_head, k_head, headSize);

    const scale = Math.sqrt(headSize) || 1;
    const scaled = scores.map((r) => r.map((s) => s / scale));

    const attn = this.applyAttentionSoftmax(scaled);

    const output = this.computeAttentionWeightedSum(attn, v_head, headSize);

    return { output, attn };
  },

  /**
   * Performs forward pass for attention mechanism.
   * Business rule: Attention mechanism allows the model to focus on different parts of the input.
   *
   * @param {Object} context - Layer context
   * @param {Float32Array} input - Input array
   * @param {number} numHeads - Number of attention heads
   * @returns {Float32Array} Attention output
   */
  attentionForward: function (context, input, numHeads = 2) {
    const validation = this.validateAttentionForwardInputs(input, numHeads);
    if (!validation) {
      return input;
    }

    const { input: validatedInput, headSize } = validation;
    const inputDim = validatedInput.length;

    if (inputDim === 0) return new Float32Array(0);

    const output = new Float32Array(inputDim).fill(0);
    const allAttentionWeights = [];

    if (context.debug) {
      console.log(
        ` Input type=${validatedInput.constructor.name}, len=${validatedInput.length}, heads=${numHeads}, headSize=${headSize}`
      );
    }

    for (let headIndex = 0; headIndex < numHeads; headIndex++) {
      const { output: headOutput, attn } = this.processAttentionHead(
        validatedInput, 
        headIndex, 
        headSize
      );

      const start = headIndex * headSize;
      for (let i = 0; i < headSize; i++) {
        output[start + i] = headOutput[i];
      }

      allAttentionWeights.push(attn);
    }

    if (context.debug) {
      console.log(
        ` Output type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`
      );
    }

    if (context.forwardCache) {
      context.forwardCache.attentionIntermediates[
        context.forwardCache.activations.length - 1
      ] = { 
        input: validatedInput, 
        numHeads, 
        headSize, 
        attentionWeights: allAttentionWeights 
      };
    }
    return output;
  },

  /**
   * Validates attention backward inputs and cache.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} dOutput - Output gradients
   * @param {Object} cache - Attention cache
   * @returns {Object} Validation result with dInput or null
   */
  validateAttentionBackwardInputs: function (dOutput, cache) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(' dOutput not Float32Array.', dOutput);
      dOutput = new Float32Array(dOutput);
    }
    if (
      !cache ||
      !(cache.input instanceof Float32Array) ||
      !Array.isArray(cache.attentionWeights)
    ) {
      const N = dOutput?.length || 0;
      return { dInput: new Float32Array(N).fill(0) };
    }

    const { input } = cache;
    const inputDim = input.length;
    if (dOutput.length !== inputDim) {
      console.error(
        `Attn Bkwd Err: dOutput size ${dOutput.length} !== input size ${inputDim}`
      );
      return { dInput: new Float32Array(inputDim).fill(0) };
    }

    return null;
  },

  /**
   * Processes attention gradients for a single head.
   * Business rule: Each attention head is processed independently.
   *
   * @param {number} headIndex - Head index
   * @param {number} headSize - Size of each head
   * @param {Float32Array} input - Input array
   * @param {Float32Array} dOutput - Output gradients
   * @param {Array} attentionWeights - Attention weights for this head
   * @param {number} scale - Scaling factor
   * @returns {Object} Gradients for this head
   */
  processAttentionHeadBackward: function (headIndex, headSize, input, dOutput, attentionWeights, scale) {
    const start = headIndex * headSize;
    const end = start + headSize;

    const q_h = input.slice(start, end);
    const v_h = q_h;
    const k_h = q_h;
    const dO_h = dOutput.slice(start, end);

    const alpha_h = attentionWeights[headIndex];
    if (
      !alpha_h ||
      alpha_h.length !== headSize ||
      alpha_h[0]?.length !== headSize
    ) {
      console.error(`Attn Bkwd Err: Head ${headIndex} weights invalid.`);
      return { dInput: new Float32Array(headSize).fill(0) };
    }

    const dQ_h = Array(headSize).fill(0);
    const dK_h = Array(headSize).fill(0);
    const dV_h = Array(headSize).fill(0);
    const dScores_h = Array.from({ length: headSize }, () =>
      Array(headSize).fill(0)
    );
    const dAlpha_h = Array.from({ length: headSize }, () =>
      Array(headSize).fill(0)
    );

    // Calculate dV and dAlpha
    this.calculateAttentionValueGradients(headSize, dO_h, v_h, alpha_h, dV_h, dAlpha_h);

    // Calculate dScores
    this.calculateAttentionScoreGradients(headSize, alpha_h, dAlpha_h, dScores_h, scale);

    // Calculate dQ and dK
    this.calculateAttentionQueryKeyGradients(headSize, dScores_h, k_h, q_h, dQ_h, dK_h);

    // Combine gradients
    const dInput = new Float32Array(headSize);
    for (let i = 0; i < headSize; i++) {
      const dQi =
        typeof dQ_h[i] === 'number' && isFinite(dQ_h[i]) ? dQ_h[i] : 0;
      const dKi =
        typeof dK_h[i] === 'number' && isFinite(dK_h[i]) ? dK_h[i] : 0;
      const dVi =
        typeof dV_h[i] === 'number' && isFinite(dV_h[i]) ? dV_h[i] : 0;
      dInput[i] = dQi + dKi + dVi;
    }

    return { dInput, start };
  },

  /**
   * Calculates value and alpha gradients for attention.
   * Business rule: Value gradients depend on attention weights.
   *
   * @param {number} headSize - Size of each head
   * @param {Float32Array} dO_h - Output gradients for this head
   * @param {Float32Array} v_h - Value array for this head
   * @param {Array} alpha_h - Attention weights for this head
   * @param {Array} dV_h - Value gradients (output)
   * @param {Array} dAlpha_h - Alpha gradients (output)
   */
  calculateAttentionValueGradients: function (headSize, dO_h, v_h, alpha_h, dV_h, dAlpha_h) {
    for (let j = 0; j < headSize; j++) {
      for (let i = 0; i < headSize; i++) {
        const dOut_i = dO_h[i];
        const val_j = v_h[j];
        const weight_ij = alpha_h[i]?.[j];

        if (
          typeof dOut_i !== 'number' ||
          !isFinite(dOut_i) ||
          typeof val_j !== 'number' ||
          !isFinite(val_j) ||
          typeof weight_ij !== 'number' ||
          !isFinite(weight_ij)
        ) {
          continue;
        }

        dV_h[j] += weight_ij * dOut_i;
        dAlpha_h[i][j] = dOut_i * val_j;
      }
    }
  },

  /**
   * Calculates score gradients for attention.
   * Business rule: Score gradients depend on alpha gradients and scaling.
   *
   * @param {number} headSize - Size of each head
   * @param {Array} alpha_h - Attention weights for this head
   * @param {Array} dAlpha_h - Alpha gradients for this head
   * @param {Array} dScores_h - Score gradients (output)
   * @param {number} scale - Scaling factor
   */
  calculateAttentionScoreGradients: function (headSize, alpha_h, dAlpha_h, dScores_h, scale) {
    for (let i = 0; i < headSize; i++) {
      let row_sum = 0;
      for (let k = 0; k < headSize; k++) {
        const dAlpha_ik = dAlpha_h[i]?.[k];
        const alpha_ik = alpha_h[i]?.[k];
        if (
          typeof dAlpha_ik === 'number' &&
          typeof alpha_ik === 'number' &&
          isFinite(dAlpha_ik) &&
          isFinite(alpha_ik)
        ) {
          row_sum += dAlpha_ik * alpha_ik;
        }
      }

      for (let j = 0; j < headSize; j++) {
        const alpha_ij = alpha_h[i]?.[j];
        const dAlpha_ij = dAlpha_h[i]?.[j];
        if (
          typeof alpha_ij === 'number' &&
          typeof dAlpha_ij === 'number' &&
          isFinite(alpha_ij) &&
          isFinite(dAlpha_ij)
        ) {
          const dS_ij = alpha_ij * (dAlpha_ij - row_sum);
          dScores_h[i][j] = dS_ij / scale;
        }
      }
    }
  },

  /**
   * Calculates query and key gradients for attention.
   * Business rule: Query and key gradients depend on score gradients.
   *
   * @param {number} headSize - Size of each head
   * @param {Array} dScores_h - Score gradients for this head
   * @param {Float32Array} k_h - Key array for this head
   * @param {Float32Array} q_h - Query array for this head
   * @param {Array} dQ_h - Query gradients (output)
   * @param {Array} dK_h - Key gradients (output)
   */
  calculateAttentionQueryKeyGradients: function (headSize, dScores_h, k_h, q_h, dQ_h, dK_h) {
    for (let i = 0; i < headSize; i++) {
      for (let j = 0; j < headSize; j++) {
        const dS_ij = dScores_h[i]?.[j];
        const k_val_j = k_h[j];
        const q_val_i = q_h[i];

        if (
          typeof dS_ij === 'number' &&
          isFinite(dS_ij) &&
          typeof k_val_j === 'number' &&
          isFinite(k_val_j) &&
          typeof q_val_i === 'number' &&
          isFinite(q_val_i)
        ) {
          dQ_h[i] += dS_ij * k_val_j;
          dK_h[j] += dS_ij * q_val_i;
        }
      }
    }
  },

  /**
   * Performs backward pass for attention mechanism.
   * Business rule: Attention backward pass computes gradients for all heads.
   *
   * @param {Object} context - Layer context
   * @param {Float32Array} dOutput - Output gradients
   * @param {Object} cache - Attention cache
   * @returns {Object} Input gradients
   */
  attentionBackward: function (context, dOutput, cache) {
    const validationResult = this.validateAttentionBackwardInputs(dOutput, cache);
    if (validationResult) {
      return validationResult;
    }

    const { input, numHeads, headSize, attentionWeights } = cache;
    const inputDim = input.length;
    const dInput = new Float32Array(inputDim).fill(0);
    const scale = Math.sqrt(headSize) || 1;

    if (context.debug) {
      console.log(
        ` Input type=${input.constructor.name}, len=${inputDim}, dOutput type=${dOutput.constructor.name}, heads=${numHeads}, headSize=${headSize}`
      );
    }

    for (let headIndex = 0; headIndex < numHeads; headIndex++) {
      const headResult = this.processAttentionHeadBackward(
        headIndex, 
        headSize, 
        input, 
        dOutput, 
        attentionWeights, 
        scale
      );
      
      if (headResult.dInput) {
        for (let i = 0; i < headSize; i++) {
          dInput[headResult.start + i] = headResult.dInput[i];
        }
      }
    }

    if (context.debug) {
      console.log(
        ` Output dInput type=${dInput.constructor.name}, len=${dInput.length}, first val=${dInput[0]?.toFixed(4)}`
      );
    }
    return { dInput };
  },



  /**
   * Performs forward pass for layer normalization.
   * Business rule: Layer normalization must be numerically stable and efficient.
   *
   * @param {Object} context - Context object with epsilon and debug settings
   * @param {Float32Array} input - Input array
   * @param {Float32Array} gamma - Gamma parameter array
   * @param {Float32Array} beta - Beta parameter array
   * @returns {Object} Layer normalization result
   */
  /**
   * Validates layer normalization inputs.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} input - Input array
   * @param {Float32Array} gamma - Gamma array
   * @param {Float32Array} beta - Beta array
   * @returns {Object} Validated inputs
   */
  validateLayerNormInputs: function (input, gamma, beta) {
    if (!(input instanceof Float32Array)) {
      console.warn(' Input not Float32Array.', input);
      input = new Float32Array(input);
    }
    if (!(gamma instanceof Float32Array)) {
      console.warn(' Gamma not Float32Array.', gamma);
      gamma = new Float32Array(gamma);
    }
    if (!(beta instanceof Float32Array)) {
      console.warn(' Beta not Float32Array.', beta);
      beta = new Float32Array(beta);
    }
    return { input, gamma, beta };
  },

  /**
   * Handles empty layer normalization input.
   * Business rule: Empty inputs should return empty outputs.
   *
   * @param {number} epsilon - Epsilon value
   * @returns {Object} Empty layer norm result
   */
  handleEmptyLayerNormInput: function (epsilon) {
    return {
      output: new Float32Array(0),
      mean: 0,
      variance: 0,
      stddev: epsilon,
      normalizedInput: new Float32Array(0)
    };
  },

  /**
   * Calculates mean and variance for layer normalization.
   * Business rule: Statistics are calculated over the entire input.
   *
   * @param {Float32Array} input - Input array
   * @param {number} N - Input length
   * @returns {Object} Mean and variance
   */
  calculateLayerNormStatistics: function (input, N) {
    let mean = 0;
    for (let i = 0; i < N; i++) {
      mean += input[i];
    }
    mean /= N;

    let variance = 0;
    for (let i = 0; i < N; i++) {
      variance += Math.pow(input[i] - mean, 2);
    }
    variance /= N;

    return { mean, variance };
  },

  /**
   * Normalizes input using layer normalization.
   * Business rule: Normalization applies gamma and beta scaling.
   *
   * @param {Float32Array} input - Input array
   * @param {Float32Array} gamma - Gamma array
   * @param {Float32Array} beta - Beta array
   * @param {number} mean - Mean value
   * @param {number} invStddev - Inverse standard deviation
   * @param {number} N - Input length
   * @returns {Object} Normalized input and output
   */
  normalizeInput: function (input, gamma, beta, mean, invStddev, N) {
    const normalizedInput = new Float32Array(N);
    const output = new Float32Array(N);

    if (gamma.length !== N || beta.length !== N) {
      console.error(
        `LN size mismatch: Input ${N}, Gamma ${gamma.length}, Beta ${beta.length}`
      );
    }

    for (let i = 0; i < N; i++) {
      normalizedInput[i] = (input[i] - mean) * invStddev;

      const g = gamma[i] ?? 1.0;
      const b = beta[i] ?? 0.0;
      output[i] = g * normalizedInput[i] + b;
    }

    return { normalizedInput, output };
  },

  /**
   * Performs forward pass for layer normalization.
   * Business rule: Layer normalization normalizes inputs and applies learnable parameters.
   *
   * @param {Object} context - Layer context
   * @param {Float32Array} input - Input array
   * @param {Float32Array} gamma - Gamma array
   * @param {Float32Array} beta - Beta array
   * @returns {Object} Layer normalization result
   */
  layerNormForward: function (context, input, gamma, beta) {
    const { input: validatedInput, gamma: validatedGamma, beta: validatedBeta } = 
      this.validateLayerNormInputs(input, gamma, beta);

    const epsilon = context.epsilon;
    const N = validatedInput.length;
    
    if (N === 0) {
      return this.handleEmptyLayerNormInput(epsilon);
    }

    const { mean, variance } = this.calculateLayerNormStatistics(validatedInput, N);
    const stddev = Math.sqrt(variance + epsilon);
    const invStddev = 1 / stddev;

    const { normalizedInput, output } = this.normalizeInput(
      validatedInput, 
      validatedGamma, 
      validatedBeta, 
      mean, 
      invStddev, 
      N
    );

    if (context.debug) {
      console.log(
        ` Input type=${validatedInput.constructor.name}, len=${N}, Mean=${mean.toFixed(4)}, Var=${variance.toFixed(4)}, StdDev=${stddev.toFixed(4)}`
      );
      console.log(
        ` Output type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`
      );
    }

    const cacheData = {
      output,
      mean,
      variance,
      stddev,
      normalizedInput,
      input: validatedInput,
      gamma: validatedGamma
    };
    
    if (context.forwardCache) {
      context.forwardCache.layerNormIntermediates[
        context.forwardCache.activations.length - 1
      ] = cacheData;
    }
    
    return cacheData;
  },

  /**
   * Validates layer normalization backward inputs.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} dOutput - Output gradients
   * @param {Object} cache - Layer normalization cache
   * @returns {Object} Validation result with gradients or null
   */
  validateLayerNormBackwardInputs: function (dOutput, cache) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(' dOutput not Float32Array.', dOutput);
      dOutput = new Float32Array(dOutput);
    }
    if (
      !cache ||
      !(cache.input instanceof Float32Array) ||
      !(cache.normalizedInput instanceof Float32Array) ||
      !(cache.gamma instanceof Float32Array)
    ) {
      const N = dOutput?.length || cache?.input?.length || 0;
      return {
        dInput: new Float32Array(N).fill(0),
        dGamma: new Float32Array(N).fill(0),
        dBeta: new Float32Array(N).fill(0)
      };
    }
    return null;
  },

  /**
   * Handles empty layer normalization backward input.
   * Business rule: Empty inputs should return empty gradients.
   *
   * @returns {Object} Empty gradients
   */
  handleEmptyLayerNormBackward: function () {
    return {
      dInput: new Float32Array(0),
      dGamma: new Float32Array(0),
      dBeta: new Float32Array(0)
    };
  },

  /**
   * Validates layer normalization backward size consistency.
   * Business rule: All arrays must have consistent dimensions.
   *
   * @param {number} N - Expected size
   * @param {Float32Array} dOutput - Output gradients
   * @param {Float32Array} normalizedInput - Normalized input
   * @param {Float32Array} gamma - Gamma array
   * @returns {Object} Validation result with gradients or null
   */
  validateLayerNormBackwardSizes: function (N, dOutput, normalizedInput, gamma) {
    if (
      dOutput.length !== N ||
      normalizedInput.length !== N ||
      gamma.length !== N
    ) {
      console.error(
        `LN Bkwd Err: Size mismatch N=${N}, dOut=${dOutput.length}, normIn=${normalizedInput.length}, gamma=${gamma.length}`
      );
      return {
        dInput: new Float32Array(N).fill(0),
        dGamma: new Float32Array(N).fill(0),
        dBeta: new Float32Array(N).fill(0)
      };
    }
    return null;
  },

  /**
   * Calculates parameter gradients for layer normalization.
   * Business rule: Parameter gradients depend on output gradients and normalized input.
   *
   * @param {Float32Array} dOutput - Output gradients
   * @param {Float32Array} normalizedInput - Normalized input
   * @param {Float32Array} gamma - Gamma array
   * @param {number} N - Array size
   * @returns {Object} Parameter gradients
   */
  calculateLayerNormParameterGradients: function (dOutput, normalizedInput, gamma, N) {
    const dGamma = new Float32Array(N);
    const dBeta = new Float32Array(N);
    const dNorm = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      dGamma[i] = dOutput[i] * normalizedInput[i];
      dBeta[i] = dOutput[i]; // dBeta gradient is just the output gradient
      dNorm[i] = dOutput[i] * (gamma[i] ?? 1);
    }

    return { dGamma, dBeta, dNorm };
  },

  /**
   * Calculates variance gradient for layer normalization.
   * Business rule: Variance gradient depends on normalized gradients and input statistics.
   *
   * @param {Float32Array} dNorm - Normalized gradients
   * @param {Float32Array} input - Input array
   * @param {number} mean - Mean value
   * @param {number} variance - Variance value
   * @param {number} epsilon - Epsilon value
   * @param {number} N - Array size
   * @returns {number} Variance gradient
   */
  calculateLayerNormVarianceGradient: function (dNorm, input, mean, variance, epsilon, N) {
    let dVariance = 0;
    for (let i = 0; i < N; i++) {
      if (isFinite(dNorm[i]) && isFinite(input[i]) && isFinite(mean)) {
        dVariance += dNorm[i] * (input[i] - mean);
      }
    }
    dVariance *= -0.5 * Math.pow(variance + epsilon, -1.5);
    return dVariance;
  },

  /**
   * Calculates mean gradient for layer normalization.
   * Business rule: Mean gradient has two components from different terms.
   *
   * @param {Float32Array} dNorm - Normalized gradients
   * @param {Float32Array} input - Input array
   * @param {number} mean - Mean value
   * @param {number} invStddev - Inverse standard deviation
   * @param {number} dVariance - Variance gradient
   * @param {number} N - Array size
   * @returns {number} Mean gradient
   */
  calculateLayerNormMeanGradient: function (dNorm, input, mean, invStddev, dVariance, N) {
    let dMean1 = 0;
    for (let i = 0; i < N; i++) {
      if (isFinite(dNorm[i])) {
        dMean1 -= dNorm[i] * invStddev;
      }
    }

    let dMean2Term = 0;
    for (let i = 0; i < N; i++) {
      if (isFinite(input[i]) && isFinite(mean)) {
        dMean2Term += -2 * (input[i] - mean);
      }
    }
    const dMean2 = (dVariance * dMean2Term) / N;
    return dMean1 + dMean2;
  },

  /**
   * Calculates input gradients for layer normalization.
   * Business rule: Input gradient has three terms from the chain rule.
   *
   * @param {Float32Array} dNorm - Normalized gradients
   * @param {Float32Array} input - Input array
   * @param {number} mean - Mean value
   * @param {number} invStddev - Inverse standard deviation
   * @param {number} dVariance - Variance gradient
   * @param {number} dMean - Mean gradient
   * @param {number} N - Array size
   * @returns {Float32Array} Input gradients
   */
  calculateLayerNormInputGradients: function (dNorm, input, mean, invStddev, dVariance, dMean, N) {
    const dInput = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const term1 = isFinite(dNorm[i]) ? dNorm[i] * invStddev : 0;
      const term2 =
        isFinite(dVariance) && isFinite(input[i]) && isFinite(mean)
          ? (dVariance * 2 * (input[i] - mean)) / N
          : 0;
      const term3 = isFinite(dMean) ? dMean / N : 0;
      dInput[i] = term1 + term2 + term3;
    }
    return dInput;
  },

  /**
   * Performs backward pass for layer normalization.
   * Business rule: Layer normalization backward pass computes gradients for input and parameters.
   *
   * @param {Object} context - Layer context
   * @param {Float32Array} dOutput - Output gradients
   * @param {Object} cache - Layer normalization cache
   * @returns {Object} Input and parameter gradients
   */
  layerNormBackward: function (context, dOutput, cache) {
    const validationResult = this.validateLayerNormBackwardInputs(dOutput, cache);
    if (validationResult) {
      return validationResult;
    }

    const { input, normalizedInput, mean, variance, stddev, gamma } = cache;
    const N = input.length;

    if (N === 0) {
      return this.handleEmptyLayerNormBackward();
    }

    const sizeValidation = this.validateLayerNormBackwardSizes(N, dOutput, normalizedInput, gamma);
    if (sizeValidation) {
      return sizeValidation;
    }

    const epsilon = context.epsilon || 1e-8;
    const invStddev = stddev > 0 ? 1 / stddev : 0;

    if (context.debug) {
      console.log(
        ` Input type=${input.constructor.name}, len=${N}, dOutput type=${dOutput.constructor.name}`
      );
    }

    const { dGamma, dBeta, dNorm } = this.calculateLayerNormParameterGradients(
      dOutput, 
      normalizedInput, 
      gamma, 
      N
    );

    const dVariance = this.calculateLayerNormVarianceGradient(
      dNorm, 
      input, 
      mean, 
      variance, 
      epsilon, 
      N
    );

    const dMean = this.calculateLayerNormMeanGradient(
      dNorm, 
      input, 
      mean, 
      invStddev, 
      dVariance, 
      N
    );

    const dInput = this.calculateLayerNormInputGradients(
      dNorm, 
      input, 
      mean, 
      invStddev, 
      dVariance, 
      dMean, 
      N
    );

    if (context.debug) {
      console.log(
        ` Output dInput type=${dInput.constructor.name}, len=${dInput.length}, first val=${dInput[0]?.toFixed(4)}`
      );
    }

    return { dInput, dGamma, dBeta };
  },

  /**
   * Validates dropout forward inputs.
   * Business rule: Input validation prevents runtime errors.
   *
   * @param {Float32Array} input - Input array
   * @returns {Float32Array} Validated input
   */
  validateDropoutInput: function (input) {
    if (!(input instanceof Float32Array)) {
      console.warn(' Input not Float32Array.', input);
      input = new Float32Array(input);
    }
    return input;
  },

  /**
   * Handles dropout when not training or rate is zero.
   * Business rule: No dropout should be applied during inference or when rate is zero.
   *
   * @param {Object} context - Layer context
   * @param {Float32Array} input - Input array
   * @param {number} idx - Layer index
   * @returns {Float32Array} Input passed through unchanged
   */
  handleDropoutNoTraining: function (context, input, idx) {
    if (!context.masks) {
      context.masks = [];
    }
    context.masks[idx] = null;
    if (context.debug) {
      console.log(
        ` Not training or rate=0. Output type=${input.constructor.name}, len=${input.length}`
      );
    }
    return input;
  },

  /**
   * Validates dropout rate.
   * Business rule: Dropout rate must be between 0 and 1.
   *
   * @param {number} rate - Dropout rate
   * @returns {boolean} True if rate is valid
   */
  validateDropoutRate: function (rate) {
    if (rate < 0 || rate >= 1) {
      console.warn(`Dropout rate ${rate} invalid`);
      return false;
    }
    return true;
  },

  /**
   * Applies dropout mask to input.
   * Business rule: Dropout randomly zeros inputs and scales remaining values.
   *
   * @param {Float32Array} input - Input array
   * @param {number} rate - Dropout rate
   * @param {number} scale - Scaling factor
   * @param {Function} randomFillFn - Random number generator
   * @returns {Object} Mask and output arrays
   */
  applyDropoutMask: function (input, rate, scale, randomFillFn) {
    const N = input.length;
    const mask = new Float32Array(N);
    const output = new Float32Array(N);

    const randInts = new Uint32Array(N);
    oblixUtils.fillRandomInts(randInts, randomFillFn);

    const threshold = rate * 4294967296;
    for (let i = 0; i < N; i++) {
      if (randInts[i] >= threshold) {
        mask[i] = scale;
        output[i] = input[i] * scale;
      } else {
        mask[i] = 0;
        output[i] = 0;
      }
    }

    return { mask, output };
  },

  /**
   * Performs forward pass for dropout layer.
   * Business rule: Dropout randomly deactivates neurons during training to prevent overfitting.
   *
   * @param {Object} context - Layer context
   * @param {Float32Array} input - Input array
   * @param {number} rate - Dropout rate
   * @returns {Float32Array} Output with dropout applied
   */
  dropoutForward: function (context, input, rate) {
    const validatedInput = this.validateDropoutInput(input);

    const idx = (context.forwardCache?.activations?.length || 1) - 1;
    
    if (!context.isTraining || rate === 0) {
      return this.handleDropoutNoTraining(context, validatedInput, idx);
    }

    if (!this.validateDropoutRate(rate)) {
      if (!context.masks) {
        context.masks = [];
      }
      context.masks[idx] = null;
      return validatedInput;
    }

    const N = validatedInput.length;
    const scale = 1 / (1 - rate);

    if (context.debug) {
      console.log(
        ` Input type=${validatedInput.constructor.name}, len=${N}, Rate=${rate}, Scale=${scale.toFixed(4)}`
      );
    }

    const { mask, output } = this.applyDropoutMask(
      validatedInput, 
      rate, 
      scale, 
      context.randomFillFn
    );
    
    if (!context.masks) {
      context.masks = [];
    }
    context.masks[idx] = mask;

    if (context.debug) {
      console.log(
        ` Output type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`
      );
    }

    return output;
  },

  dropoutBackward: function (context, dOutput, layerIndex) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(' dOutput not Float32Array.', dOutput);
      dOutput = new Float32Array(dOutput);
    }

    const mask = context.masks[layerIndex];

    if (!mask) {
      if (context.debug)
        console.log(
          ` L${layerIndex}: No mask found (not training or rate=0). Passing through dOutput.`
        );
      return dOutput;
    }

    if (!(mask instanceof Float32Array)) {
      console.error(` L${layerIndex}: Mask is not a Float32Array!`, mask);
      return dOutput;
    }

    if (dOutput.length !== mask.length) {
      console.error(
        ` L${layerIndex}: dOutput size ${dOutput.length} !== mask size ${mask.length}`
      );

      return dOutput;
    }

    const dInput = new Float32Array(dOutput.length);
    for (let i = 0; i < dInput.length; i++) {
      const grad = dOutput[i];
      const maskVal = mask[i];
      if (isFinite(grad) && isFinite(maskVal)) {
        dInput[i] = grad * maskVal;
      } else {
        dInput[i] = 0;
      }
    }

    if (context.debug)
      console.log(
        ` L${layerIndex}: Applied mask. dInput type=${dInput.constructor.name}, len=${dInput.length}, first val=${dInput[0]?.toFixed(4)}`
      );

    return dInput;
  },

  softmaxForward: function (context, input) {
    if (!(input instanceof Float32Array)) {
      console.warn(': Input not Float32Array.', input);
      input = new Float32Array(input);
    }

    const N = input.length;
    if (N === 0) return new Float32Array(0);

    let maxVal = -Infinity;
    for (let i = 0; i < N; i++) {
      if (input[i] > maxVal) maxVal = input[i];
    }

    const exps = new Float32Array(N);
    let sumExps = 0;
    for (let i = 0; i < N; i++) {
      const expVal = Math.exp(input[i] - maxVal);
      exps[i] = expVal;
      sumExps += expVal;
    }

    sumExps += 1e-9;

    const output = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      output[i] = exps[i] / sumExps;
    }

    if (context.debug)
      console.log(
        `: Input type=${input.constructor.name}, len=${N}, Output type=${output.constructor.name}, first val=${output[0]?.toFixed(4)}`
      );

    if (context.forwardCache)
      context.forwardCache.softmaxOutputs[
        context.forwardCache.activations.length - 1
      ] = output;
    return output;
  },

  softmaxBackward: function (context, dOutput, layerIndex) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(' dOutput not Float32Array.', dOutput);
      dOutput = new Float32Array(dOutput);
    }

    if (context.debug)
      console.log(
        ` L${layerIndex}: Passing through gradient (assuming CE Loss). dInput type=${dOutput.constructor.name}, len=${dOutput.length}`
      );

    return dOutput;
  }
};

