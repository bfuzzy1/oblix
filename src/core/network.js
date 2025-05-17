const oblixActivations = {
  apply: function (x, activation) {
    const alpha = 0.01;
    const softplus = (v) => Math.log(1 + Math.exp(v));
    switch (activation) {
      case "tanh":
        return Math.tanh(x);
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "relu":
        return Math.max(0, x);
      case "leakyrelu":
        return x > 0 ? x : alpha * x;
      case "gelu":
        return (
          0.5 *
          x *
          (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
        );
      case "selu":
        const sa = 1.67326,
          ss = 1.0507;
        return x > 0 ? ss * x : ss * sa * (Math.exp(x) - 1);
      case "swish":
        return x / (1 + Math.exp(-x));
      case "mish":
        return x * Math.tanh(softplus(x));
      case "softmax":
      case "none":
      default:
        return x;
    }
  },
  derivative: function (x, activation) {
    const alpha = 0.01;
    let val;
    const sigmoid = (v) => 1 / (1 + Math.exp(-v));
    const softplus = (v) => Math.log(1 + Math.exp(v));
    const dtanh_dx = (v) => 1 - Math.tanh(v) ** 2;
    switch (activation) {
      case "tanh":
        val = Math.tanh(x);
        return 1 - val * val;
      case "sigmoid":
        val = sigmoid(x);
        return val * (1 - val);
      case "relu":
        return x > 0 ? 1 : 0;
      case "leakyrelu":
        return x > 0 ? 1 : alpha;
      case "gelu":
        const k = Math.sqrt(2 / Math.PI),
          inner = k * (x + 0.044715 * x ** 3),
          tanh_inner = Math.tanh(inner),
          d_inner_dx = k * (1 + 0.134145 * x ** 2),
          sech_sq_inner = 1 - tanh_inner ** 2;
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech_sq_inner * d_inner_dx;
      case "selu":
        const sa = 1.67326,
          ss = 1.0507;
        return x > 0 ? ss : ss * sa * Math.exp(x);
      case "swish":
        const sig_x = sigmoid(x);
        return sig_x + x * sig_x * (1 - sig_x);
      case "mish":
        const sp_x = softplus(x);
        const tanh_sp_x = Math.tanh(sp_x);
        const dsp_dx = sigmoid(x);
        const dtanh_dsp = dtanh_dx(sp_x);
        return tanh_sp_x + x * dtanh_dsp * dsp_dx;
      case "softmax":
      case "none":
      default:
        return 1;
    }
  },
};

const oblixLayerOps = {
  attentionForward: function (context, input, numHeads = 2) {
    if (!(input instanceof Float32Array)) {
      console.warn(" Input not Float32Array.", input);
      input = new Float32Array(input);
    }
    const inputDim = input.length;
    if (inputDim === 0) return new Float32Array(0);
    if (
      numHeads <= 0 ||
      !Number.isInteger(numHeads) ||
      inputDim % numHeads !== 0
    ) {
      console.error(`Attn Error: Dim ${inputDim} not divisible by ${numHeads}`);
      return input;
    }

    const headSize = inputDim / numHeads;

    const output = new Float32Array(inputDim).fill(0);
    const allAttentionWeights = [];

    if (context.debug)
      console.log(
        ` Input type=${input.constructor.name}, len=${input.length}, heads=${numHeads}, headSize=${headSize}`,
      );

    for (let h = 0; h < numHeads; h++) {
      const start = h * headSize;
      const end = start + headSize;

      const q_head = input.slice(start, end);
      const k_head = q_head;
      const v_head = q_head;

      const scores = Array.from({ length: headSize }, (_, i) =>
        Array.from({ length: headSize }, (_, j) => q_head[i] * k_head[j]),
      );

      const scale = Math.sqrt(headSize) || 1;
      const scaled = scores.map((r) => r.map((s) => s / scale));

      const attn = scaled.map((r) => {
        if (r.length === 0) return [];
        const max = Math.max(...r, -Infinity);
        const exps = r.map((s) => Math.exp(s - max));
        const sum = exps.reduce((a, b) => a + b, 1e-9);
        return exps.map((e) => e / sum);
      });
      allAttentionWeights.push(attn);

      for (let i = 0; i < headSize; i++) {
        let weighted_sum = 0;
        for (let j = 0; j < headSize; j++) {
          weighted_sum += attn[i][j] * v_head[j];
        }
        output[start + i] = weighted_sum;
      }
    }

    if (context.debug)
      console.log(
        ` Output type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`,
      );

    if (context.forwardCache)
      context.forwardCache.attentionIntermediates[
        context.forwardCache.activations.length - 1
      ] = { input, numHeads, headSize, attentionWeights: allAttentionWeights };
    return output;
  },

  attentionBackward: function (context, dOutput, cache) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(" dOutput not Float32Array.", dOutput);
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

    const { input, numHeads, headSize, attentionWeights } = cache;
    const inputDim = input.length;
    if (dOutput.length !== inputDim) {
      console.error(
        `Attn Bkwd Err: dOutput size ${dOutput.length} !== input size ${inputDim}`,
      );
      return { dInput: new Float32Array(inputDim).fill(0) };
    }

    const dInput = new Float32Array(inputDim).fill(0);
    const scale = Math.sqrt(headSize) || 1;

    if (context.debug)
      console.log(
        ` Input type=${input.constructor.name}, len=${inputDim}, dOutput type=${dOutput.constructor.name}, heads=${numHeads}, headSize=${headSize}`,
      );

    for (let h = 0; h < numHeads; h++) {
      const start = h * headSize;
      const end = start + headSize;

      const q_h = input.slice(start, end);
      const v_h = q_h;
      const k_h = q_h;
      const dO_h = dOutput.slice(start, end);

      const alpha_h = attentionWeights[h];
      if (
        !alpha_h ||
        alpha_h.length !== headSize ||
        alpha_h[0]?.length !== headSize
      ) {
        console.error(`Attn Bkwd Err: Head ${h} weights invalid.`);
        continue;
      }

      const dQ_h = Array(headSize).fill(0);
      const dK_h = Array(headSize).fill(0);
      const dV_h = Array(headSize).fill(0);
      const dScores_h = Array.from({ length: headSize }, () =>
        Array(headSize).fill(0),
      );
      const dAlpha_h = Array.from({ length: headSize }, () =>
        Array(headSize).fill(0),
      );

      for (let j = 0; j < headSize; j++) {
        for (let i = 0; i < headSize; i++) {
          const dOut_i = dO_h[i];
          const val_j = v_h[j];
          const weight_ij = alpha_h[i]?.[j];

          if (
            typeof dOut_i !== "number" ||
            !isFinite(dOut_i) ||
            typeof val_j !== "number" ||
            !isFinite(val_j) ||
            typeof weight_ij !== "number" ||
            !isFinite(weight_ij)
          ) {
            continue;
          }

          dV_h[j] += weight_ij * dOut_i;
          dAlpha_h[i][j] = dOut_i * val_j;
        }
      }

      for (let i = 0; i < headSize; i++) {
        let row_sum = 0;
        for (let k = 0; k < headSize; k++) {
          const dAlpha_ik = dAlpha_h[i]?.[k];
          const alpha_ik = alpha_h[i]?.[k];
          if (
            typeof dAlpha_ik === "number" &&
            typeof alpha_ik === "number" &&
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
            typeof alpha_ij === "number" &&
            typeof dAlpha_ij === "number" &&
            isFinite(alpha_ij) &&
            isFinite(dAlpha_ij)
          ) {
            const dS_ij = alpha_ij * (dAlpha_ij - row_sum);
            dScores_h[i][j] = dS_ij / scale;
          }
        }
      }

      for (let i = 0; i < headSize; i++) {
        for (let j = 0; j < headSize; j++) {
          const dS_ij = dScores_h[i]?.[j];
          const k_val_j = k_h[j];
          const q_val_i = q_h[i];

          if (
            typeof dS_ij === "number" &&
            isFinite(dS_ij) &&
            typeof k_val_j === "number" &&
            isFinite(k_val_j) &&
            typeof q_val_i === "number" &&
            isFinite(q_val_i)
          ) {
            dQ_h[i] += dS_ij * k_val_j;
            dK_h[j] += dS_ij * q_val_i;
          }
        }
      }

      for (let i = 0; i < headSize; i++) {
        const dQi =
          typeof dQ_h[i] === "number" && isFinite(dQ_h[i]) ? dQ_h[i] : 0;
        const dKi =
          typeof dK_h[i] === "number" && isFinite(dK_h[i]) ? dK_h[i] : 0;
        const dVi =
          typeof dV_h[i] === "number" && isFinite(dV_h[i]) ? dV_h[i] : 0;
        dInput[start + i] = dQi + dKi + dVi;
      }
    }
    if (context.debug)
      console.log(
        ` Output dInput type=${dInput.constructor.name}, len=${dInput.length}, first val=${dInput[0]?.toFixed(4)}`,
      );
    return { dInput };
  },

  layerNormForward: function (context, input, gamma, beta) {
    if (!(input instanceof Float32Array)) {
      console.warn(" Input not Float32Array.", input);
      input = new Float32Array(input);
    }
    if (!(gamma instanceof Float32Array)) {
      console.warn(" Gamma not Float32Array.", gamma);
      gamma = new Float32Array(gamma);
    }
    if (!(beta instanceof Float32Array)) {
      console.warn(" Beta not Float32Array.", beta);
      beta = new Float32Array(beta);
    }

    const epsilon = context.epsilon;
    const N = input.length;
    if (N === 0)
      return {
        output: new Float32Array(0),
        mean: 0,
        variance: 0,
        stddev: epsilon,
        normalizedInput: new Float32Array(0),
      };

    let mean = 0;
    for (let i = 0; i < N; i++) mean += input[i];
    mean /= N;

    let variance = 0;
    for (let i = 0; i < N; i++) variance += Math.pow(input[i] - mean, 2);
    variance /= N;

    const stddev = Math.sqrt(variance + epsilon);
    const invStddev = 1 / stddev;

    const normalizedInput = new Float32Array(N);
    const output = new Float32Array(N);

    if (gamma.length !== N || beta.length !== N)
      console.error(
        `LN size mismatch: Input ${N}, Gamma ${gamma.length}, Beta ${beta.length}`,
      );

    if (context.debug)
      console.log(
        ` Input type=${input.constructor.name}, len=${N}, Mean=${mean.toFixed(4)}, Var=${variance.toFixed(4)}, StdDev=${stddev.toFixed(4)}`,
      );

    for (let i = 0; i < N; i++) {
      normalizedInput[i] = (input[i] - mean) * invStddev;

      const g = gamma[i] ?? 1.0;
      const b = beta[i] ?? 0.0;
      output[i] = g * normalizedInput[i] + b;
    }

    if (context.debug)
      console.log(
        ` Output type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`,
      );

    const cacheData = {
      output,
      mean,
      variance,
      stddev,
      normalizedInput,
      input,
      gamma,
    };
    if (context.forwardCache)
      context.forwardCache.layerNormIntermediates[
        context.forwardCache.activations.length - 1
      ] = cacheData;
    return cacheData;
  },

  layerNormBackward: function (context, dOutput, cache) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(" dOutput not Float32Array.", dOutput);
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
        dBeta: new Float32Array(N).fill(0),
      };
    }

    const { input, normalizedInput, mean, variance, stddev, gamma } = cache;
    const N = input.length;

    if (N === 0)
      return {
        dInput: new Float32Array(0),
        dGamma: new Float32Array(0),
        dBeta: new Float32Array(0),
      };
    if (
      dOutput.length !== N ||
      normalizedInput.length !== N ||
      gamma.length !== N
    ) {
      console.error(
        `LN Bkwd Err: Size mismatch N=${N}, dOut=${dOutput.length}, normIn=${normalizedInput.length}, gamma=${gamma.length}`,
      );
      return {
        dInput: new Float32Array(N).fill(0),
        dGamma: new Float32Array(N).fill(0),
        dBeta: new Float32Array(N).fill(0),
      };
    }

    const epsilon = context.epsilon;
    const invStddev = 1 / stddev;

    const dGamma = new Float32Array(N);
    const dBeta = new Float32Array(dOutput);
    const dNorm = new Float32Array(N);

    if (context.debug)
      console.log(
        ` Input type=${input.constructor.name}, len=${N}, dOutput type=${dOutput.constructor.name}`,
      );

    for (let i = 0; i < N; i++) {
      dGamma[i] = dOutput[i] * normalizedInput[i];
      dNorm[i] = dOutput[i] * (gamma[i] ?? 1);
    }

    let dVariance = 0;
    for (let i = 0; i < N; i++) {
      if (isFinite(dNorm[i]) && isFinite(input[i]) && isFinite(mean)) {
        dVariance += dNorm[i] * (input[i] - mean);
      }
    }
    dVariance *= -0.5 * Math.pow(variance + epsilon, -1.5);

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
    const dMean = dMean1 + dMean2;

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

    if (context.debug)
      console.log(
        ` Output dInput type=${dInput.constructor.name}, len=${dInput.length}, first val=${dInput[0]?.toFixed(4)}`,
      );

    return { dInput, dGamma, dBeta };
  },

  dropoutForward: function (context, input, rate) {
    if (!(input instanceof Float32Array)) {
      console.warn(" Input not Float32Array.", input);
      input = new Float32Array(input);
    }

    const idx = (context.forwardCache?.activations?.length || 1) - 1;
    if (!context.isTraining || rate === 0) {
      context.masks[idx] = null;
      if (context.debug)
        console.log(
          ` Not training or rate=0. Output type=${input.constructor.name}, len=${input.length}`,
        );
      return input;
    }
    if (rate < 0 || rate >= 1) {
      console.warn(`Dropout rate ${rate} invalid`);
      context.masks[idx] = null;
      return input;
    }

    const N = input.length;
    const scale = 1 / (1 - rate);

    const mask = new Float32Array(N);
    const output = new Float32Array(N);

    if (context.debug)
      console.log(
        ` Input type=${input.constructor.name}, len=${N}, Rate=${rate}, Scale=${scale.toFixed(4)}`,
      );

    for (let i = 0; i < N; i++) {
      if (Math.random() > rate) {
        mask[i] = scale;
        output[i] = input[i] * scale;
      } else {
        mask[i] = 0;
        output[i] = 0;
      }
    }
    context.masks[idx] = mask;

    if (context.debug)
      console.log(
        ` Output type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`,
      );

    return output;
  },

  dropoutBackward: function (context, dOutput, layerIndex) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(" dOutput not Float32Array.", dOutput);
      dOutput = new Float32Array(dOutput);
    }

    const mask = context.masks[layerIndex];

    if (!mask) {
      if (context.debug)
        console.log(
          ` L${layerIndex}: No mask found (not training or rate=0). Passing through dOutput.`,
        );
      return dOutput;
    }

    if (!(mask instanceof Float32Array)) {
      console.error(` L${layerIndex}: Mask is not a Float32Array!`, mask);
      return dOutput;
    }

    if (dOutput.length !== mask.length) {
      console.error(
        ` L${layerIndex}: dOutput size ${dOutput.length} !== mask size ${mask.length}`,
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
        ` L${layerIndex}: Applied mask. dInput type=${dInput.constructor.name}, len=${dInput.length}, first val=${dInput[0]?.toFixed(4)}`,
      );

    return dInput;
  },

  softmaxForward: function (context, input) {
    if (!(input instanceof Float32Array)) {
      console.warn(": Input not Float32Array.", input);
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
        `: Input type=${input.constructor.name}, len=${N}, Output type=${output.constructor.name}, first val=${output[0]?.toFixed(4)}`,
      );

    if (context.forwardCache)
      context.forwardCache.softmaxOutputs[
        context.forwardCache.activations.length - 1
      ] = output;
    return output;
  },

  softmaxBackward: function (context, dOutput, layerIndex) {
    if (!(dOutput instanceof Float32Array)) {
      console.warn(" dOutput not Float32Array.", dOutput);
      dOutput = new Float32Array(dOutput);
    }

    if (context.debug)
      console.log(
        ` L${layerIndex}: Passing through gradient (assuming CE Loss). dInput type=${dOutput.constructor.name}, len=${dOutput.length}`,
      );

    return dOutput;
  },
};

const oblixOptimizers = {
  initializeState: function (context, optimizer) {
    const numLayers = context.layers.length;
    if (context.debug)
      console.log(`Init optimizer state: ${optimizer}, ${numLayers} layers.`);
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

    for (let i = 0; i < numLayers; i++) {
      const cfg = context.layers[i];
      if (!cfg) continue;
      const w = context.weights[i];
      const reqW =
        cfg.type === "dense" &&
        Array.isArray(w) &&
        w.length > 0 &&
        Array.isArray(w[0]);
      const b = context.biases[i];
      const reqB =
        cfg.type === "dense" && cfg.useBias && Array.isArray(b) && b.length > 0;
      const g = context.gammas[i];
      const beta = context.betas[i];
      const reqLN =
        cfg.type === "layernorm" &&
        Array.isArray(g) &&
        Array.isArray(beta) &&
        g.length === beta.length &&
        g.length > 0;

      if (
        optimizer === "adam" ||
        optimizer === "rmsprop" ||
        optimizer === "adamw"
      ) {
        if (reqW) {
          try {
            const z = () => w.map((r) => r.map(() => 0));
            if (optimizer === "adam" || optimizer === "adamw") {
              context.m_dw[i] = z();
              context.v_dw[i] = z();
            }
            if (optimizer === "rmsprop") {
              context.s_dw[i] = z();
            }
          } catch (e) {
            console.error(`InitOpt L${i} W err: ${e.message}`);
            context.m_dw[i] = null;
            context.v_dw[i] = null;
            context.s_dw[i] = null;
          }
        }
        if (reqB) {
          try {
            const z = () => b.map(() => 0);
            if (optimizer === "adam" || optimizer === "adamw") {
              context.m_db[i] = z();
              context.v_db[i] = z();
            }
            if (optimizer === "rmsprop") {
              context.s_db[i] = z();
            }
          } catch (e) {
            console.error(`InitOpt L${i} B err: ${e.message}`);
            context.m_db[i] = null;
            context.v_db[i] = null;
            context.s_db[i] = null;
          }
        }
        if (reqLN) {
          try {
            const z = () => g.map(() => 0);
            if (optimizer === "adam" || optimizer === "adamw") {
              context.m_dgamma[i] = z();
              context.v_dgamma[i] = z();
              context.m_dbeta[i] = z();
              context.v_dbeta[i] = z();
            }
            if (optimizer === "rmsprop") {
              context.s_dgamma[i] = z();
              context.s_dbeta[i] = z();
            }
          } catch (e) {
            console.error(`InitOpt L${i} LN err: ${e.message}`);
            context.m_dgamma[i] = null;
            context.v_dgamma[i] = null;
            context.m_dbeta[i] = null;
            context.v_dbeta[i] = null;
            context.s_dgamma[i] = null;
            context.s_dbeta[i] = null;
          }
        }
      }
    }
    if (context.debug) console.log(`Optimizer state init finished.`);
  },

  updateParameters: function (
    context,
    gradsW,
    gradsB,
    gradsGamma,
    gradsBeta,
    options,
  ) {
    const {
      learningRate,
      initialLearningRate,
      optimizer,
      batchSize,
      l2Lambda,
      gradientClipValue,
      decayRate,
    } = options;

    const batchMult = batchSize > 0 ? 1.0 / batchSize : 1.0;
    context.t++;

    if (context.debug)
      console.log(
        ` T=${context.t}, LR=${learningRate.toExponential(3)}, BatchSize=${batchSize}, Opt=${optimizer}, L2=${l2Lambda}`,
      );

    for (let i = 0; i < context.layers.length; i++) {
      const cfg = context.layers[i];
      const isDense = cfg.type === "dense";
      const isLN = cfg.type === "layernorm";

      let stepLR = learningRate;
      let adamCorrectedBaseLR =
        (initialLearningRate * Math.sqrt(1 - context.beta2 ** context.t)) /
        (1 - context.beta1 ** context.t);
      let adamStepLR =
        adamCorrectedBaseLR * (learningRate / initialLearningRate);

      const applyUpdate = (
        param,
        grad,
        mState,
        vState,
        sState,
        paramIdx,
        layerIdx,
        paramType,
      ) => {
        if (
          typeof param !== "number" ||
          typeof grad !== "number" ||
          !isFinite(param) ||
          !isFinite(grad)
        ) {
          return param;
        }

        const effectiveGrad = grad * batchMult;
        const clippedGrad =
          gradientClipValue > 0
            ? Math.max(
                -gradientClipValue,
                Math.min(gradientClipValue, effectiveGrad),
              )
            : effectiveGrad;
        if (!isFinite(clippedGrad)) {
          return param;
        }

        let update = 0;
        let m = typeof mState === "number" && isFinite(mState) ? mState : 0;
        let v = typeof vState === "number" && isFinite(vState) ? vState : 0;
        let s = typeof sState === "number" && isFinite(sState) ? sState : 0;

        if (optimizer === "adam" || optimizer === "adamw") {
          m = context.beta1 * m + (1 - context.beta1) * clippedGrad;
          v = context.beta2 * v + (1 - context.beta2) * clippedGrad ** 2;
          const m_hat = m / (1 - context.beta1 ** context.t);
          const v_hat = v / (1 - context.beta2 ** context.t);

          const sqrt_v_hat = Math.sqrt(v_hat);
          if (
            isFinite(m_hat) &&
            isFinite(sqrt_v_hat) &&
            sqrt_v_hat + context.epsilon !== 0
          ) {
            update = (adamStepLR * m_hat) / (sqrt_v_hat + context.epsilon);
          } else {
            update = 0;
          }

          if (optimizer === "adam" && l2Lambda > 0) {
            update += adamStepLR * l2Lambda * param;
          }
        } else if (optimizer === "rmsprop") {
          s = decayRate * s + (1 - decayRate) * clippedGrad ** 2;
          const sqrt_s = Math.sqrt(s);
          if (
            isFinite(clippedGrad) &&
            isFinite(sqrt_s) &&
            sqrt_s + context.epsilon !== 0
          ) {
            update = (stepLR * clippedGrad) / (sqrt_s + context.epsilon);
          } else {
            update = 0;
          }
          if (l2Lambda > 0) {
            update += stepLR * l2Lambda * param;
          }
        } else {
          update = stepLR * clippedGrad;
          if (l2Lambda > 0) {
            update += stepLR * l2Lambda * param;
          }
        }

        if (!isFinite(update)) {
          return { param, m, v, s };
        }

        param -= update;

        if (optimizer === "adamw" && l2Lambda > 0 && paramType === "W") {
          param -= adamStepLR * l2Lambda * param;
        }

        if (!isFinite(param)) {
          return { param: NaN, m, v, s };
        }

        return { param, m, v, s };
      };

      if (
        isDense &&
        context.weights[i] instanceof Float32Array &&
        gradsW[i] instanceof Float32Array
      ) {
        const w = context.weights[i];
        const gW = gradsW[i];
        const mW = context.m_dw[i];
        const vW = context.v_dw[i];
        const sW = context.s_dw[i];

        for (let k = 0; k < w.length; k++) {
          const mVal = mW ? mW[k] : null;
          const vVal = vW ? vW[k] : null;
          const sVal = sW ? sW[k] : null;

          const result = applyUpdate(w[k], gW[k], mVal, vVal, sVal, k, i, "W");

          if (isNaN(result.param)) {
            console.warn(
              `Optimizer L${i}-W[${k}] resulted in NaN. Keeping original value ${w[k]}.`,
            );
          } else {
            w[k] = result.param;
            if (mW) mW[k] = result.m;
            if (vW) vW[k] = result.v;
            if (sW) sW[k] = result.s;
          }
        }
        if (context.debug && i === 0 && w.length > 0)
          console.log(
            ` Opt L0 W: First few updated = ${w.slice(0, 3).map((v) => v.toFixed(4))}`,
          );
      }

      if (
        isDense &&
        cfg.useBias &&
        context.biases[i] instanceof Float32Array &&
        gradsB[i] instanceof Float32Array
      ) {
        const b = context.biases[i];
        const gB = gradsB[i];
        const mB = context.m_db[i];
        const vB = context.v_db[i];
        const sB = context.s_db[i];

        for (let k = 0; k < b.length; k++) {
          const mVal = mB ? mB[k] : null;
          const vVal = vB ? vB[k] : null;
          const sVal = sB ? sB[k] : null;

          const result = applyUpdate(b[k], gB[k], mVal, vVal, sVal, k, i, "B");

          if (isNaN(result.param)) {
            console.warn(
              `Optimizer L${i}-B[${k}] resulted in NaN. Keeping original value ${b[k]}.`,
            );
          } else {
            b[k] = result.param;
            if (mB) mB[k] = result.m;
            if (vB) vB[k] = result.v;
            if (sB) sB[k] = result.s;
          }
        }
        if (context.debug && i === 0 && b.length > 0)
          console.log(
            ` Opt L0 B: First few updated = ${b.slice(0, 3).map((v) => v.toFixed(4))}`,
          );
      }

      if (
        isLN &&
        context.gammas[i] instanceof Float32Array &&
        context.betas[i] instanceof Float32Array &&
        gradsGamma[i] instanceof Float32Array &&
        gradsBeta[i] instanceof Float32Array
      ) {
        const gamma = context.gammas[i];
        const gGamma = gradsGamma[i];
        const beta = context.betas[i];
        const gBeta = gradsBeta[i];
        const mGamma = context.m_dgamma[i];
        const vGamma = context.v_dgamma[i];
        const sGamma = context.s_dgamma[i];
        const mBeta = context.m_dbeta[i];
        const vBeta = context.v_dbeta[i];
        const sBeta = context.s_dbeta[i];

        for (let k = 0; k < gamma.length; k++) {
          const mGammaVal = mGamma ? mGamma[k] : null;
          const vGammaVal = vGamma ? vGamma[k] : null;
          const sGammaVal = sGamma ? sGamma[k] : null;
          const resultGamma = applyUpdate(
            gamma[k],
            gGamma[k],
            mGammaVal,
            vGammaVal,
            sGammaVal,
            k,
            i,
            "Gamma",
          );

          if (isNaN(resultGamma.param)) {
            console.warn(
              `Optimizer L${i}-Gamma[${k}] resulted in NaN. Keeping original value ${gamma[k]}.`,
            );
          } else {
            gamma[k] = resultGamma.param;
            if (mGamma) mGamma[k] = resultGamma.m;
            if (vGamma) vGamma[k] = resultGamma.v;
            if (sGamma) sGamma[k] = resultGamma.s;
          }

          const mBetaVal = mBeta ? mBeta[k] : null;
          const vBetaVal = vBeta ? vBeta[k] : null;
          const sBetaVal = sBeta ? sBeta[k] : null;
          const resultBeta = applyUpdate(
            beta[k],
            gBeta[k],
            mBetaVal,
            vBetaVal,
            sBetaVal,
            k,
            i,
            "Beta",
          );

          if (isNaN(resultBeta.param)) {
            console.warn(
              `Optimizer L${i}-Beta[${k}] resulted in NaN. Keeping original value ${beta[k]}.`,
            );
          } else {
            beta[k] = resultBeta.param;
            if (mBeta) mBeta[k] = resultBeta.m;
            if (vBeta) vBeta[k] = resultBeta.v;
            if (sBeta) sBeta[k] = resultBeta.s;
          }
        }
        if (context.debug && i < 2 && gamma.length > 0)
          console.log(
            ` Opt L${i} LN: First Gamma=${gamma[0]?.toFixed(4)}, Beta=${beta[0]?.toFixed(4)}`,
          );
      }
    }
  },
};

const oblixUtils = {
  positionalEncoding: function (input, maxLen = -1) {
    const dModel = input.length;
    if (dModel === 0) return new Float32Array(0);

    let inputArray;
    if (input instanceof Float32Array) {
      inputArray = input;
    } else if (Array.isArray(input)) {
      console.warn(" Input was standard Array, converting to Float32Array.");
      inputArray = new Float32Array(input);
    } else {
      console.error(" Invalid input type.", input);
      return new Float32Array(dModel);
    }

    if (maxLen < 0) maxLen = dModel;
    const pe = new Float32Array(dModel).fill(0);

    for (let i = 0; i < dModel; i++) {
      const position = i;
      const divTermBase = Math.pow(10000, (Math.floor(i / 2) * 2) / dModel);
      if (divTermBase === 0) continue;

      const angle = position / divTermBase;
      pe[i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }

    const output = new Float32Array(dModel);
    for (let i = 0; i < dModel; i++) {
      output[i] = inputArray[i] + pe[i];
    }

    if (
      input.constructor &&
      output.constructor &&
      typeof console !== "undefined" &&
      console.log
    ) {
      if (console.debug) {
        console.debug(
          ` Input type=${input.constructor.name}, Output type=${output.constructor.name}, len=${output.length}`,
        );
      } else {
      }
    }

    return output;
  },

  calculateAccuracy: function (predictions, targets) {
    if (
      !predictions ||
      !targets ||
      predictions.length === 0 ||
      predictions.length !== targets.length
    ) {
      console.warn("Accuracy calculation: Invalid input arrays.");
      return 0.0;
    }

    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
      const predVec = predictions[i];
      const targetInfo = targets[i];

      if (!predVec || !targetInfo) continue;

      let predictedIndex = -1;
      let maxPredVal = -Infinity;
      for (let j = 0; j < predVec.length; j++) {
        const v = predVec[j];
        if (typeof v === "number" && isFinite(v) && v > maxPredVal) {
          maxPredVal = v;
          predictedIndex = j;
        }
      }

      let targetIndex = -1;

      if (typeof targetInfo === "number" && Number.isInteger(targetInfo)) {
        targetIndex = targetInfo;
      } else if (
        Array.isArray(targetInfo) &&
        targetInfo.length === predVec.length
      ) {
        for (let j = 0; j < targetInfo.length; j++) {
          const t = targetInfo[j];
          if (t === 1) {
            targetIndex = j;
            break;
          }
          if (targetIndex === -1 && typeof t === "number" && isFinite(t)) {
            if (j === 0 || t > targetInfo[targetIndex]) {
              targetIndex = j;
            }
          }
        }
      } else {
        console.warn(
          `Accuracy calc: Invalid target format at index ${i}`,
          targetInfo,
        );
        continue;
      }

      if (predictedIndex === targetIndex && targetIndex !== -1) {
        correct++;
      }
    }
    return predictions.length > 0 ? correct / predictions.length : 0.0;
  },

  calculateRSquared: function (predictions, targets) {
    if (
      !predictions ||
      !targets ||
      predictions.length === 0 ||
      predictions.length !== targets.length
    ) {
      console.warn("R-squared calculation: Invalid input arrays.");
      return NaN;
    }
    if (predictions.length < 2) {
      console.warn("R-squared calculation: Need at least 2 data points.");
      return NaN;
    }

    const targetMean =
      targets.reduce((sum, val) => sum + val, 0) / targets.length;
    if (typeof console !== "undefined" && console.log) {
      console.log(
        `R² Func Check: Received preds[0]=${predictions[0]} (typeof: ${typeof predictions[0]}), targets[0]=${targets[0]} (typeof: ${typeof targets[0]})`,
      );
      console.log(`R² Func Check: Calculated targetMean = ${targetMean}`);
    }

    let ssTot = 0;
    let ssRes = 0;

    for (let i = 0; i < targets.length; i++) {
      const targetVal = targets[i];
      const predVal = predictions[i];

      if (typeof console !== "undefined" && console.log) {
        const isTargetNum = typeof targetVal === "number";
        const isPredNum = typeof predVal === "number";
        const isTargetFinite = isFinite(targetVal);
        const isPredFinite = isFinite(predVal);
        if (i === 0) {
          console.log(
            `R² Func Loop i=${i}: Checks -> targetNum=${isTargetNum}, predNum=${isPredNum}, targetFinite=${isTargetFinite}, predFinite=${isPredFinite}`,
          );
        }
      }

      if (!isFinite(targetVal) || !isFinite(predVal)) {
        console.warn(
          `R-squared calc: Non-finite number at index ${i} (Pred: ${predVal}, Target: ${targetVal})`,
        );
        return NaN;
      }

      ssTot += (targetVal - targetMean) ** 2;
      ssRes += (targetVal - predVal) ** 2;
    }

    if (ssTot === 0) {
      return ssRes === 0 ? 1.0 : NaN;
    }
    return 1 - ssRes / ssTot;
  },

  _gaussian_spare: null,
  gaussianRandom: function () {
    if (this._gaussian_spare !== null) {
      const spare = this._gaussian_spare;
      this._gaussian_spare = null;
      return spare;
    }
    let u, v, s;
    do {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
    this._gaussian_spare = v * mul;
    return u * mul;
  },

  generateXORData: function (numSamples, noise) {
    const data = [];
    for (let i = 0; i < numSamples; i++) {
      const x1 = Math.random() < 0.5 ? 0 : 1;
      const x2 = Math.random() < 0.5 ? 0 : 1;
      const raw_output = (x1 + x2) % 2;
      const noisy_x1 = x1 + oblixUtils.gaussianRandom() * noise * 0.5;
      const noisy_x2 = x2 + oblixUtils.gaussianRandom() * noise * 0.5;
      data.push({ input: [noisy_x1, noisy_x2], output: [raw_output] });
    }
    return data;
  },

  generateLinearData: function (
    numSamples,
    noise,
    inputDims = 1,
    outputDims = 1,
  ) {
    const data = [];
    const weights = Array.from(
      { length: inputDims },
      () => (Math.random() - 0.5) * 2,
    );
    const bias = Math.random() - 0.5;

    for (let i = 0; i < numSamples; i++) {
      const input = Array.from(
        { length: inputDims },
        () => Math.random() * 2 - 1,
      );
      let linearCombination = bias;
      for (let d = 0; d < inputDims; d++) {
        linearCombination += input[d] * weights[d];
      }

      const output = Array.from({ length: outputDims }, (_, idx) => {
        return (
          linearCombination * (1 + idx * 0.05) +
          oblixUtils.gaussianRandom() * noise
        );
      });
      data.push({ input: input, output: output });
    }
    return data;
  },

  generateCircularData: function (numSamples, noise) {
    const data = [];
    const radiusSeparation = 2;
    const numInner = Math.floor(numSamples / 2);
    const numOuter = numSamples - numInner;

    for (let i = 0; i < numInner; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * radiusSeparation * 0.4;
      const x = radius * Math.cos(angle) + oblixUtils.gaussianRandom() * noise;
      const y = radius * Math.sin(angle) + oblixUtils.gaussianRandom() * noise;
      data.push({ input: [x, y], output: [0] });
    }

    for (let i = 0; i < numOuter; i++) {
      const angle = Math.random() * 2 * Math.PI;

      const radius =
        radiusSeparation * 0.7 + Math.random() * radiusSeparation * 0.4;
      const x = radius * Math.cos(angle) + oblixUtils.gaussianRandom() * noise;
      const y = radius * Math.sin(angle) + oblixUtils.gaussianRandom() * noise;
      data.push({ input: [x, y], output: [1] });
    }
    return data;
  },

  generateGaussianBlobs: function (numSamples, noise, numClasses = 2) {
    const data = [];
    const centers = [];
    const centerSpread = 4;
    for (let c = 0; c < numClasses; c++) {
      centers.push([
        (Math.random() - 0.5) * centerSpread,
        (Math.random() - 0.5) * centerSpread,
      ]);
    }

    for (let i = 0; i < numSamples; i++) {
      const classIndex = Math.floor(Math.random() * numClasses);
      const center = centers[classIndex];

      const stdDev = noise + 0.5;
      const x = center[0] + oblixUtils.gaussianRandom() * stdDev;
      const y = center[1] + oblixUtils.gaussianRandom() * stdDev;

      const output = [classIndex];
      data.push({ input: [x, y], output: output });
    }
    return data;
  },
};

export class Oblix {
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

    if (this.debug)
      console.log(
        "oblix instance created. Initializing for Float32Array storage...",
      );
  }

  reset() {
    if (this.debug) console.log("Resetting oblix instance...");
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
    if (this.debug) console.log("Oblix reset complete.");
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
    if (typeof inputSize !== "number" || inputSize <= 0)
      throw new Error(
        `Layer ${this.layers.length}: Invalid inputSize: ${inputSize}.`,
      );
    if (this.layers.length > 0) {
      const prevLayer = this.layers[this.layers.length - 1];
      if (inputSize !== prevLayer.outputSize)
        throw new Error(
          `Layer ${this.layers.length} (${type}): Input size ${inputSize} doesn't match previous layer's output size ${prevLayer.outputSize}.`,
        );
    }
    let actualOutputSize = outputSize;
    switch (type) {
      case "dense":
        if (typeof outputSize !== "number" || outputSize <= 0)
          throw new Error(
            `Dense Layer ${this.layers.length}: Invalid outputSize: ${outputSize}.`,
          );
        break;
      case "layernorm":
      case "attention":
      case "dropout":
      case "softmax":
        actualOutputSize = inputSize;
        if (outputSize !== undefined && outputSize !== inputSize)
          console.warn(
            `${type} layer ${this.layers.length}: Output size ignored.`,
          );
        break;
      default:
        throw new Error(`Unknown layer type: ${type}`);
    }
    if (type === "attention" && inputSize % numHeads !== 0)
      throw new Error(
        `Attention layer ${this.layers.length}: Input size ${inputSize} not divisible by numHeads ${numHeads}.`,
      );
    if (type === "dropout" && (rate < 0 || rate >= 1))
      throw new Error(
        `Dropout layer ${this.layers.length}: Rate ${rate} must be >= 0 and < 1.`,
      );

    const layerConfig = {
      type,
      inputSize,
      outputSize: actualOutputSize,
      activation,
      numHeads,
      useBias,
      rate,
      weightInit,
    };
    const layerIndex = this.layers.length;
    this.layers.push(layerConfig);

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

    if (type === "dense") {
      const weightCount = actualOutputSize * inputSize;
      const weightsArray = new Float32Array(weightCount);
      let initFunc;
      if (weightInit === "he") {
        const stdDev = Math.sqrt(2 / inputSize);
        initFunc = () => oblixUtils.gaussianRandom() * stdDev;
        if (this.debug)
          console.log(
            `L${layerIndex} Dense Weights init: He (stdDev=${stdDev.toFixed(4)})`,
          );
      } else {
        const limit = Math.sqrt(6 / (inputSize + actualOutputSize));
        initFunc = () => (Math.random() * 2 - 1) * limit;
        if (this.debug)
          console.log(
            `L${layerIndex} Dense Weights init: Glorot (limit=${limit.toFixed(4)})`,
          );
      }

      for (let i = 0; i < weightCount; i++) {
        weightsArray[i] = initFunc();
      }
      this.weights[layerIndex] = weightsArray;

      if (useBias) {
        this.biases[layerIndex] = new Float32Array(actualOutputSize).fill(0.01);
        if (this.debug)
          console.log(
            `L${layerIndex} Dense Biases init: ${this.biases[layerIndex] instanceof Float32Array}, Length: ${this.biases[layerIndex]?.length}`,
          );
      }
    } else if (type === "layernorm") {
      this.gammas[layerIndex] = new Float32Array(actualOutputSize).fill(1.0);
      this.betas[layerIndex] = new Float32Array(actualOutputSize).fill(0.0);
      if (this.debug)
        console.log(
          `L${layerIndex} LayerNorm Gamma init: ${this.gammas[layerIndex] instanceof Float32Array}, Length: ${this.gammas[layerIndex]?.length}`,
        );
      if (this.debug)
        console.log(
          `L${layerIndex} LayerNorm Beta init: ${this.betas[layerIndex] instanceof Float32Array}, Length: ${this.betas[layerIndex]?.length}`,
        );
    }
  }

  initializeOptimizerState(optimizer) {
    const numLayers = this.layers.length;
    if (this.debug)
      console.log(
        `Init optimizer state (${optimizer}) for ${numLayers} layers. Creating Float32Arrays.`,
      );
    this.t = 0;

    const ensureLen = (arrName) => {
      if (!this[arrName] || this[arrName].length !== numLayers) {
        this[arrName] = Array(numLayers).fill(null);
      }
    };
    [
      "m_dw",
      "v_dw",
      "m_db",
      "v_db",
      "m_dgamma",
      "v_dgamma",
      "m_dbeta",
      "v_dbeta",
      "s_dw",
      "s_db",
      "s_dgamma",
      "s_dbeta",
    ].forEach(ensureLen);

    for (let i = 0; i < numLayers; i++) {
      const cfg = this.layers[i];
      if (!cfg) continue;
      const w = this.weights[i];
      const b = this.biases[i];
      const g = this.gammas[i];
      const beta = this.betas[i];

      const needsWState = cfg.type === "dense" && w instanceof Float32Array;
      const needsBState =
        cfg.type === "dense" && cfg.useBias && b instanceof Float32Array;
      const needsLNState =
        cfg.type === "layernorm" &&
        g instanceof Float32Array &&
        beta instanceof Float32Array;

      if (
        optimizer === "adam" ||
        optimizer === "rmsprop" ||
        optimizer === "adamw"
      ) {
        try {
          if (needsWState) {
            const size = w.length;
            if (optimizer === "adam" || optimizer === "adamw") {
              if (!this.m_dw[i]) this.m_dw[i] = new Float32Array(size).fill(0);
              if (!this.v_dw[i]) this.v_dw[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === "rmsprop") {
              if (!this.s_dw[i]) this.s_dw[i] = new Float32Array(size).fill(0);
            }

            if (this.debug)
              console.log(
                `L${i} Dense W Opt State (${optimizer}) init: ${this.m_dw[i]?.length || this.s_dw[i]?.length} elements`,
              );
          }
          if (needsBState) {
            const size = b.length;
            if (optimizer === "adam" || optimizer === "adamw") {
              if (!this.m_db[i]) this.m_db[i] = new Float32Array(size).fill(0);
              if (!this.v_db[i]) this.v_db[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === "rmsprop") {
              if (!this.s_db[i]) this.s_db[i] = new Float32Array(size).fill(0);
            }
            if (this.debug)
              console.log(
                `L${i} Dense B Opt State (${optimizer}) init: ${this.m_db[i]?.length || this.s_db[i]?.length} elements`,
              );
          }
          if (needsLNState) {
            const size = g.length;
            if (optimizer === "adam" || optimizer === "adamw") {
              if (!this.m_dgamma[i])
                this.m_dgamma[i] = new Float32Array(size).fill(0);
              if (!this.v_dgamma[i])
                this.v_dgamma[i] = new Float32Array(size).fill(0);
              if (!this.m_dbeta[i])
                this.m_dbeta[i] = new Float32Array(size).fill(0);
              if (!this.v_dbeta[i])
                this.v_dbeta[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === "rmsprop") {
              if (!this.s_dgamma[i])
                this.s_dgamma[i] = new Float32Array(size).fill(0);
              if (!this.s_dbeta[i])
                this.s_dbeta[i] = new Float32Array(size).fill(0);
            }
            if (this.debug)
              console.log(
                `L${i} LN Opt State (${optimizer}) init: ${this.m_dgamma[i]?.length || this.s_dgamma[i]?.length} elements`,
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
    if (this.debug) console.log(`Optimizer state init finished.`);
  }

  getTotalParameters() {
    let total = 0;
    if (!this.layers || this.layers.length === 0) return 0;
    this.layers.forEach((l, i) => {
      if (l.type === "dense") {
        total += this.weights[i] ? this.weights[i].length : 0;
        total += l.useBias && this.biases[i] ? this.biases[i].length : 0;
      } else if (l.type === "layernorm") {
        total += this.gammas[i] ? this.gammas[i].length : 0;
        total += this.betas[i] ? this.betas[i].length : 0;
      }
    });

    if (this.debug)
      console.log(` getTotalParameters: Calculated total: ${total}`);
    return total;
  }

  getCurrentLearningRate(epoch, initialLR, options) {
    const { lrSchedule, lrStepDecayFactor, lrStepDecaySize, lrExpDecayRate } =
      options;
    let currentLR = initialLR;

    if (lrSchedule === "step") {
      const stepSize = Math.max(1, Math.floor(lrStepDecaySize));

      const decaySteps = Math.floor(epoch / stepSize);
      currentLR = initialLR * Math.pow(lrStepDecayFactor, decaySteps);
    } else if (lrSchedule === "exponential") {
      currentLR = initialLR * Math.pow(lrExpDecayRate, epoch);
    }

    return currentLR;
  }

  async train(trainSet, options = {}) {
    this.isTraining = true;
    const start = Date.now();
    let {
      epochs = 100,
      learningRate = 0.01,
      batchSize = 16,
      printEveryEpochs = 10,
      earlyStopThreshold = 1e-7,
      testSet = null,
      callback = null,
      optimizer = "adam",
      lossFunction = "mse",
      l2Lambda = 0,
      decayRate = 0.9,
      usePositionalEncoding = this.usePositionalEncoding,
      gradientClipValue = 0,
      lrSchedule = "none",
      lrStepDecayFactor = 0.1,
      lrStepDecaySize = 10,
      lrExpDecayRate = 0.95,
    } = options;

    const initialLearningRate = learningRate;

    if (!trainSet || trainSet.length === 0)
      throw new Error("Training set empty.");
    if (this.layers.length === 0) throw new Error("No layers.");
    const effectiveBatchSize = Math.max(
      1,
      Math.min(batchSize, trainSet.length),
    );
    this.usePositionalEncoding = usePositionalEncoding;
    this.decayRate = decayRate;
    let needsOptimizerInit =
      this.m_dw?.length !== this.layers.length ||
      this.v_dw?.length !== this.layers.length ||
      this.s_dw?.length !== this.layers.length ||
      this.m_db?.length !== this.layers.length ||
      this.v_db?.length !== this.layers.length ||
      this.s_db?.length !== this.layers.length;
    for (let i = 0; i < this.layers.length && !needsOptimizerInit; i++) {
      if (this.layers[i].type === "dense") {
        if (this.weights[i] && this.m_dw[i] === null) needsOptimizerInit = true;
        if (this.biases[i] && this.m_db[i] === null) needsOptimizerInit = true;
      } else if (this.layers[i].type === "layernorm") {
        if (this.gammas[i] && this.m_dgamma[i] === null)
          needsOptimizerInit = true;
        if (this.betas[i] && this.m_dbeta[i] === null)
          needsOptimizerInit = true;
      }
    }
    if (needsOptimizerInit) {
      if (this.debug) console.log("Optimizer state needs init.");
      this.initializeOptimizerState(optimizer);
    }
    let lastTrainLoss = Infinity;
    let lastTestLoss = null;

    let lastValidationMetric = null;
    let validationMetricName = "";

    for (let epoch = 0; epoch < epochs; epoch++) {
      while (this.isPaused) {
        await new Promise((r) => setTimeout(r, 50));
      }
      const currentEpochLearningRate = this.getCurrentLearningRate(
        epoch,
        initialLearningRate,
        options,
      );

      let totalEpochTrainError = 0;
      for (let i = trainSet.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]];
      }
      for (let b = 0; b < trainSet.length; b += effectiveBatchSize) {
        while (this.isPaused) {
          await new Promise((r) => setTimeout(r, 50));
        }
        const batch = trainSet.slice(b, b + effectiveBatchSize);
        if (batch.length === 0) continue;

        const gradsW = this.weights.map((L) =>
          L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null,
        );
        const gradsB = this.biases.map((L) =>
          L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null,
        );
        const gradsGamma = this.gammas.map((L) =>
          L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null,
        );
        const gradsBeta = this.betas.map((L) =>
          L instanceof Float32Array ? new Float32Array(L.length).fill(0) : null,
        );

        let batchLossSum = 0;
        for (const data of batch) {
          while (this.isPaused) {
            await new Promise((r) => setTimeout(r, 50));
          }
          let currentInput = data.input;
          if (!Array.isArray(currentInput) || !Array.isArray(data.output)) {
            console.warn("Skip invalid data");
            continue;
          }

          if (this.usePositionalEncoding) {
            currentInput = oblixUtils.positionalEncoding(currentInput);
          }

          const initialAct =
            currentInput instanceof Float32Array
              ? currentInput
              : new Float32Array(currentInput);
          this.forwardCache = {
            activations: [initialAct],
            rawValues: [],
            layerNormIntermediates: [],
            attentionIntermediates: [],
            softmaxOutputs: [],
          };
          let layerInput = this.forwardCache.activations[0];

          for (let i = 0; i < this.layers.length; i++) {
            const cfg = this.layers[i];
            let out;

            this.forwardCache.rawValues[i] = null;
            this.forwardCache.layerNormIntermediates[i] = null;
            this.forwardCache.attentionIntermediates[i] = null;
            this.forwardCache.softmaxOutputs[i] = null;

            try {
              if (!(layerInput instanceof Float32Array))
                throw new Error(
                  `L${i}(${cfg.type}): Internal error - input is not Float32Array.`,
                );
              if (layerInput.length !== cfg.inputSize)
                throw new Error(
                  `L${i}(${cfg.type}): Sz mismatch ${layerInput.length}!=${cfg.inputSize}`,
                );

              switch (cfg.type) {
                case "dense":
                  const w = this.weights[i];
                  const b = this.biases[i];
                  if (!(w instanceof Float32Array))
                    throw new Error(`L${i} Dense: Weights not Float32Array.`);
                  if (b && !(b instanceof Float32Array))
                    throw new Error(`L${i} Dense: Bias not Float32Array.`);

                  const rawSums = new Float32Array(cfg.outputSize);
                  for (let j = 0; j < cfg.outputSize; ++j) {
                    let sum = b ? b[j] : 0;
                    const weightRowOffset = j * cfg.inputSize;
                    for (let k = 0; k < cfg.inputSize; ++k) {
                      sum += layerInput[k] * w[weightRowOffset + k];
                    }
                    rawSums[j] = sum;
                  }
                  this.forwardCache.rawValues[i] = rawSums;

                  out = new Float32Array(cfg.outputSize);
                  for (let j = 0; j < cfg.outputSize; ++j) {
                    out[j] = oblixActivations.apply(rawSums[j], cfg.activation);
                  }

                  break;
                case "layernorm":
                  out = oblixLayerOps.layerNormForward(
                    this,
                    layerInput,
                    this.gammas[i],
                    this.betas[i],
                  ).output;
                  break;
                case "attention":
                  out = oblixLayerOps.attentionForward(
                    this,
                    layerInput,
                    cfg.numHeads,
                  );
                  break;
                case "dropout":
                  out = oblixLayerOps.dropoutForward(
                    this,
                    layerInput,
                    cfg.rate,
                  );
                  break;
                case "softmax":
                  out = oblixLayerOps.softmaxForward(this, layerInput);
                  break;
                default:
                  throw new Error(`Fwd Pass: Unknown type ${cfg.type}`);
              }

              if (!(out instanceof Float32Array))
                throw new Error(
                  `L${i}(${cfg.type}): Internal error - output is not Float32Array.`,
                );
              this.forwardCache.activations.push(out);
              layerInput = out;
            } catch (e) {
              console.error(`Fwd L${i}(${cfg.type}) Err:`, e);
              this.isTraining = false;
              throw e;
            }
          }

          const finalOut = layerInput;
          const targetOut = data.output;
          if (finalOut.length !== targetOut.length)
            throw new Error(`Output/Target len mismatch`);
          let dLastErr;
          const eps_ce = 1e-9;

          if (lossFunction === "crossentropy") {
            let loss = 0;
            const lastLyr = this.layers[this.layers.length - 1];
            const wasSoftmax =
              lastLyr.type === "softmax" ||
              (lastLyr.type === "dense" && lastLyr.activation === "softmax");
            const wasSigmoid =
              lastLyr.type === "dense" && lastLyr.activation === "sigmoid";

            if (wasSoftmax) {
              const oneHotTarget = new Float32Array(finalOut.length).fill(0);
              if (
                targetOut.length === 1 &&
                Number.isInteger(targetOut[0]) &&
                targetOut[0] >= 0 &&
                targetOut[0] < finalOut.length
              ) {
                oneHotTarget[targetOut[0]] = 1;
              } else if (targetOut.length === finalOut.length) {
                for (let i = 0; i < targetOut.length; ++i)
                  oneHotTarget[i] = targetOut[i];
              } else {
                throw new Error("CE target unclear");
              }

              for (let i = 0; i < finalOut.length; ++i)
                loss -= oneHotTarget[i] * Math.log(finalOut[i] + eps_ce);

              dLastErr = new Float32Array(finalOut.length);
              for (let i = 0; i < finalOut.length; ++i)
                dLastErr[i] = finalOut[i] - oneHotTarget[i];
            } else if (wasSigmoid) {
              if (finalOut.length !== 1 || targetOut.length !== 1)
                throw new Error("BCE needs single out/target");
              const p = finalOut[0],
                t = targetOut[0];
              loss = -(
                t * Math.log(p + eps_ce) +
                (1 - t) * Math.log(1 - p + eps_ce)
              );
              dLastErr = new Float32Array([p - t]);
            } else {
              console.warn("CE w/o final softmax/sigmoid, using simple diff");

              dLastErr = new Float32Array(finalOut.length);
              for (let i = 0; i < finalOut.length; ++i)
                dLastErr[i] = finalOut[i] - targetOut[i];

              loss = 0.5 * dLastErr.reduce((s, e) => s + e * e, 0);
            }
            if (!isNaN(loss)) batchLossSum += loss;
          } else {
            dLastErr = new Float32Array(finalOut.length);
            let loss = 0;
            for (let i = 0; i < finalOut.length; ++i) {
              const diff = finalOut[i] - targetOut[i];
              dLastErr[i] = diff;
              loss += diff * diff;
            }
            loss *= 0.5;
            if (!isNaN(loss)) batchLossSum += loss;
          }

          let dAct = dLastErr;

          for (let i = this.layers.length - 1; i >= 0; i--) {
            const cfg = this.layers[i];
            const act_prev = this.forwardCache.activations[i];
            let dIn;

            if (
              !(dAct instanceof Float32Array) ||
              dAct.length !== cfg.outputSize
            ) {
              console.warn(
                `Bkwd L${i}(${cfg.type}): Invalid dAct. Type: ${dAct?.constructor?.name}, Len: ${dAct?.length}. Expected Len: ${cfg.outputSize}. Using zeros.`,
              );

              dIn = new Float32Array(cfg.inputSize).fill(0);
              dAct = new Float32Array(cfg.outputSize).fill(0);
              continue;
            }

            try {
              switch (cfg.type) {
                case "dense":
                  const w = this.weights[i];
                  const b = this.biases[i];
                  const raw = this.forwardCache.rawValues[i];
                  const act = cfg.activation;
                  const inSz = cfg.inputSize;
                  const outSz = cfg.outputSize;

                  if (!(raw instanceof Float32Array))
                    throw new Error(
                      `L${i} Dense Bkwd: Missing or invalid raw values cache.`,
                    );
                  if (!(w instanceof Float32Array))
                    throw new Error(
                      `L${i} Dense Bkwd: Weights not Float32Array.`,
                    );
                  if (!(act_prev instanceof Float32Array))
                    throw new Error(
                      `L${i} Dense Bkwd: Previous activation not Float32Array.`,
                    );

                  const delta = new Float32Array(outSz);
                  for (let j = 0; j < outSz; ++j) {
                    const deriv = oblixActivations.derivative(raw[j], act);
                    if (typeof deriv !== "number" || !isFinite(deriv)) {
                      console.warn(
                        `L${i} Dense, j=${j}: Deriv NaN/Inf. Activation: ${act}, Raw Input: ${raw[j]}, Derivative: ${deriv}`,
                      );
                      delta[j] = 0;
                    } else {
                      delta[j] = dAct[j] * deriv;
                    }
                  }

                  dIn = new Float32Array(inSz).fill(0);
                  for (let k = 0; k < inSz; k++) {
                    for (let j = 0; j < outSz; j++) {
                      const weightIndex = j * inSz + k;
                      dIn[k] += delta[j] * w[weightIndex];
                    }
                  }

                  const gW = gradsW[i];
                  const gB = gradsB[i];
                  if (gW) {
                    for (let j = 0; j < outSz; j++) {
                      const weightRowOffset = j * inSz;
                      for (let k = 0; k < inSz; k++) {
                        gW[weightRowOffset + k] += delta[j] * act_prev[k];
                      }
                    }
                  }
                  if (gB) {
                    for (let j = 0; j < outSz; j++) {
                      gB[j] += delta[j];
                    }
                  }

                  break;
                case "layernorm":
                  const lnCache = this.forwardCache.layerNormIntermediates[i];
                  if (!lnCache) throw new Error(`L${i} LN Bkwd: Missing cache`);
                  const {
                    dInput: ln_dIn,
                    dGamma,
                    dBeta,
                  } = oblixLayerOps.layerNormBackward(this, dAct, lnCache);
                  dIn = ln_dIn;
                  const gGamma = gradsGamma[i];
                  const gBeta = gradsBeta[i];
                  if (gGamma && gBeta) {
                    for (let j = 0; j < dGamma.length; j++) {
                      gGamma[j] += dGamma[j] || 0;
                      gBeta[j] += dBeta[j] || 0;
                    }
                  }
                  break;
                case "attention":
                  const attnCache = this.forwardCache.attentionIntermediates[i];
                  if (!attnCache)
                    throw new Error(`L${i} Attn Bkwd: Missing cache`);
                  const { dInput: attn_dIn } = oblixLayerOps.attentionBackward(
                    this,
                    dAct,
                    attnCache,
                  );
                  dIn = attn_dIn;

                  break;
                case "dropout":
                  dIn = oblixLayerOps.dropoutBackward(this, dAct, i);
                  break;
                case "softmax":
                  dIn = oblixLayerOps.softmaxBackward(this, dAct, i);
                  break;
                default:
                  throw new Error(`Bkwd Pass: Unknown type ${cfg.type}`);
              }

              if (!(dIn instanceof Float32Array))
                throw new Error(
                  `Bkwd L${i}(${cfg.type}): Internal error - dIn is not Float32Array.`,
                );
              dAct = dIn;
            } catch (e) {
              console.error(`Bkwd L${i}(${cfg.type}) Err:`, e);
              this.isTraining = false;
              throw e;
            }
          }
        }

        const updateOptions = {
          learningRate: currentEpochLearningRate,
          initialLearningRate: initialLearningRate,
          optimizer: optimizer,
          batchSize: batch.length,
          l2Lambda: l2Lambda,
          gradientClipValue: gradientClipValue,
          decayRate: this.decayRate,
        };
        oblixOptimizers.updateParameters(
          this,
          gradsW,
          gradsB,
          gradsGamma,
          gradsBeta,
          updateOptions,
        );

        totalEpochTrainError += batchLossSum;
      }

      lastTrainLoss = totalEpochTrainError / trainSet.length;

      if (testSet && testSet.length > 0) {
        let testError = 0;
        const allValPredictions = [];
        const allValTargets = [];

        for (const data of testSet) {
          while (this.isPaused) {
            await new Promise((r) => setTimeout(r, 50));
          }
          const prediction = this.predict(data.input);
          if (
            prediction &&
            data.output &&
            prediction.length === data.output.length
          ) {
            const target = data.output;
            allValPredictions.push(prediction);
            allValTargets.push(target);

            let sampleLoss = 0;
            const eps_ce = 1e-9;
            if (lossFunction === "crossentropy") {
              const lastLayer = this.layers[this.layers.length - 1];
              const wasSoftmax =
                lastLayer.type === "softmax" ||
                (lastLayer.type === "dense" &&
                  lastLayer.activation === "softmax");
              const wasSigmoid =
                lastLayer.type === "dense" &&
                lastLayer.activation === "sigmoid";

              if (wasSoftmax) {
                if (
                  target.length === 1 &&
                  Number.isInteger(target[0]) &&
                  target[0] >= 0 &&
                  target[0] < prediction.length
                ) {
                  sampleLoss = -Math.log(prediction[target[0]] + eps_ce);
                } else if (target.length === prediction.length) {
                  sampleLoss = -target.reduce(
                    (sum, t, i) => sum + t * Math.log(prediction[i] + eps_ce),
                    0,
                  );
                } else {
                  console.warn(
                    "CE Val Loss: Target/Prediction shape mismatch for Softmax.",
                  );
                  sampleLoss = NaN;
                }
              } else if (
                wasSigmoid &&
                prediction.length === 1 &&
                target.length === 1
              ) {
                const p = prediction[0];
                const t = target[0];
                sampleLoss = -(
                  t * Math.log(p + eps_ce) +
                  (1 - t) * Math.log(1 - p + eps_ce)
                );
              } else {
                console.warn(
                  "CE Val Loss: Final layer activation not Softmax/Sigmoid. Using MSE for loss metric.",
                );
                sampleLoss =
                  0.5 *
                  prediction.reduce(
                    (sum, p, i) => sum + (p - target[i]) ** 2,
                    0,
                  );
              }
            } else {
              sampleLoss =
                0.5 *
                prediction.reduce((sum, p, i) => sum + (p - target[i]) ** 2, 0);
            }

            if (!isNaN(sampleLoss) && isFinite(sampleLoss)) {
              testError += sampleLoss;
            }
          }
        }
        lastTestLoss = testError / testSet.length;

        if (lossFunction === "crossentropy") {
          validationMetricName = "Acc";

          lastValidationMetric = oblixUtils.calculateAccuracy(
            allValPredictions,
            allValTargets,
          );
        } else {
          validationMetricName = "R²";

          const flatPreds = allValPredictions.flat();
          const flatTargets = allValTargets.flat();

          if (this.debug) {
            console.log(
              `R-Squared Type Check: flatPreds type = ${flatPreds?.constructor?.name}, flatTargets type = ${flatTargets?.constructor?.name}`,
            );
            console.log(
              `R-Squared Input Check: flatPreds[0]=${flatPreds[0]}, flatTargets[0]=${flatTargets[0]}`,
            );
            if (flatPreds.length > 1)
              console.log(
                `  flatPreds[1]=${flatPreds[1]}, flatTargets[1]=${flatTargets[1]}`,
              );
          }

          lastValidationMetric = oblixUtils.calculateRSquared(
            flatPreds,
            flatTargets,
          );
        }
      } else {
        lastTestLoss = null;
        lastValidationMetric = null;
        validationMetricName = "";
      }

      if ((epoch + 1) % printEveryEpochs === 0 && this.debug) {
        const lrStr =
          lrSchedule !== "none"
            ? `, LR: ${currentEpochLearningRate.toExponential(2)}`
            : "";
        let logMsg = `Epoch ${epoch + 1}/${epochs}, Train Loss: ${lastTrainLoss.toFixed(6)}`;
        if (lastTestLoss !== null) {
          logMsg += `, Val Loss: ${lastTestLoss.toFixed(6)}`;
        }

        if (lastValidationMetric !== null && !isNaN(lastValidationMetric)) {
          logMsg += `, Val ${validationMetricName}: ${lastValidationMetric.toFixed(4)}`;
        }
        logMsg += lrStr;
        console.log(logMsg);
      }

      if (callback) {
        await callback(
          epoch + 1,
          lastTrainLoss,
          lastTestLoss,
          validationMetricName,
          lastValidationMetric,
          this.forwardCache,
        );
      }

      await new Promise((resolve) => setTimeout(resolve, 0));
      if (lastTrainLoss < earlyStopThreshold) {
        if (this.debug) console.log(`Early stopping @ Epoch ${epoch + 1}.`);
        epochs = epoch + 1;
        break;
      }
    }

    const end = Date.now();
    this.isTraining = false;
    const totalParams = this.getTotalParameters();

    const trainingSummary = {
      trainLoss: lastTrainLoss,
      testLoss: lastTestLoss,
      validationMetric: {
        name: validationMetricName,
        value: lastValidationMetric,
      },
      parameters: totalParams,
      training: {
        time: end - start,
        epochs: epochs,
        learningRate: initialLearningRate,
        batchSize: effectiveBatchSize,
        optimizer: optimizer,
        lossFunction: lossFunction,
        l2Lambda: l2Lambda,
        decayRate: this.decayRate,
        usePositionalEncoding: this.usePositionalEncoding,
        gradientClipValue: gradientClipValue,
        lrSchedule: lrSchedule,
        lrStepDecayFactor:
          lrSchedule === "step" ? lrStepDecayFactor : undefined,
        lrStepDecaySize: lrSchedule === "step" ? lrStepDecaySize : undefined,
        lrExpDecayRate:
          lrSchedule === "exponential" ? lrExpDecayRate : undefined,
      },
      layers: this.layers.map((l) => ({
        type: l.type,
        inputSize: l.inputSize,
        outputSize: l.outputSize,
        activation: l.activation,
        numHeads: l.numHeads,
        useBias: l.useBias,
        rate: l.rate,
      })),
    };
    this.details = trainingSummary;
    if (this.debug) console.log("Training finished.", trainingSummary);
    return trainingSummary;
  }

  pauseTraining() {
    this.isPaused = true;
  }

  resumeTraining() {
    this.isPaused = false;
  }

  predict(input) {
    if (this.debug)
      console.log(" Starting prediction with native Float32Array logic.");

    const wasTraining = this.isTraining;
    this.isTraining = false;
    if (!this.layers || this.layers.length === 0) {
      console.error("Predict Error: Model not initialized.");
      return null;
    }
    if (!input || typeof input.length !== "number") {
      console.error("Predict Error: Invalid input provided.", input);
      return null;
    }

    let currentInput;
    if (input instanceof Float32Array) {
      currentInput = input;
    } else if (Array.isArray(input)) {
      if (this.debug)
        console.log(" Input is standard array, converting to Float32Array.");
      currentInput = new Float32Array(input);
    } else {
      console.error(
        "Predict Error: Input is not an array or Float32Array.",
        input,
      );
      return null;
    }

    if (this.debug)
      console.log(
        ` Initial Input type=${currentInput.constructor.name}, len=${currentInput.length}`,
      );

    if (this.usePositionalEncoding) {
      if (this.debug) console.log(" Applying positional encoding.");
      currentInput = oblixUtils.positionalEncoding(currentInput);
      if (this.debug)
        console.log(
          ` After PosEnc type=${currentInput.constructor.name}, len=${currentInput.length}`,
        );
    }

    this.lastActivations = [currentInput];

    try {
      for (let i = 0; i < this.layers.length; i++) {
        const cfg = this.layers[i];
        const layerInput =
          this.lastActivations[this.lastActivations.length - 1];

        if (this.debug)
          console.log(
            ` Processing L${i} (${cfg.type}). Input type=${layerInput.constructor.name}, len=${layerInput.length}`,
          );

        if (!(layerInput instanceof Float32Array)) {
          throw new Error(
            `L${i}(${cfg.type}): Internal error - input is not Float32Array.`,
          );
        }
        if (layerInput.length !== cfg.inputSize) {
          throw new Error(
            `L${i}(${cfg.type}): Size mismatch. Expected ${cfg.inputSize}, got ${layerInput.length}.`,
          );
        }

        let output;

        switch (cfg.type) {
          case "dense":
            const w = this.weights[i];
            const b = this.biases[i];
            if (!w) throw new Error(`L${i} Dense: Weights not initialized.`);
            if (!(w instanceof Float32Array))
              throw new Error(
                `L${i} Dense: Weights internal error - not Float32Array.`,
              );
            if (b && !(b instanceof Float32Array))
              throw new Error(
                `L${i} Dense: Biases internal error - not Float32Array.`,
              );

            output = new Float32Array(cfg.outputSize);
            if (this.debug)
              console.log(
                ` L${i} Dense: InputLen=${layerInput.length}, WeightLen=${w.length}, BiasLen=${b?.length}, OutputLen=${output.length}`,
              );

            for (let j = 0; j < cfg.outputSize; ++j) {
              let sum = b ? b[j] : 0;
              const weightRowOffset = j * cfg.inputSize;

              for (let k = 0; k < cfg.inputSize; ++k) {
                sum += layerInput[k] * w[weightRowOffset + k];
              }

              output[j] = oblixActivations.apply(sum, cfg.activation);

              if (this.debug && j === 0 && i < 2) {
                console.log(
                  ` L${i} Dense, Neuron 0: Sum=${sum.toFixed(4)}, Activated=${output[0].toFixed(4)}`,
                );
              }
            }

            break;

          case "layernorm":
            const gamma = this.gammas[i];
            const beta = this.betas[i];
            if (!gamma || !beta)
              throw new Error(`L${i} LayerNorm: Gamma/Beta not initialized.`);
            if (
              !(gamma instanceof Float32Array) ||
              !(beta instanceof Float32Array)
            )
              throw new Error(
                `L${i} LN: Gamma/Beta internal error - not Float32Array.`,
              );

            const { output: lnOut } = oblixLayerOps.layerNormForward(
              this,
              layerInput,
              gamma,
              beta,
            );
            output = lnOut;
            break;

          case "attention":
            output = oblixLayerOps.attentionForward(
              this,
              layerInput,
              cfg.numHeads,
            );
            break;

          case "dropout":
            output = oblixLayerOps.dropoutForward(this, layerInput, cfg.rate);
            break;

          case "softmax":
            output = oblixLayerOps.softmaxForward(this, layerInput);
            break;

          default:
            throw new Error(`Predict: Unknown layer type ${cfg.type}`);
        }

        if (!(output instanceof Float32Array)) {
          throw new Error(
            `L${i}(${cfg.type}): Internal error - output is not Float32Array.`,
          );
        }
        if (this.debug)
          console.log(
            ` Output L${i} (${cfg.type}) type=${output.constructor.name}, len=${output.length}, first val=${output[0]?.toFixed(4)}`,
          );
        this.lastActivations.push(output);
      }

      this.isTraining = wasTraining;
      const finalOutput = this.lastActivations[this.lastActivations.length - 1];
      if (this.debug)
        console.log(
          ` Finished. Final output type=${finalOutput?.constructor?.name}, len=${finalOutput?.length}`,
        );
      return finalOutput;
    } catch (error) {
      console.error("Prediction Error:", error);
      this.lastActivations = null;
      this.isTraining = wasTraining;
      return null;
    }
  }

  save(name = "model") {
    if (!this.layers || this.layers.length === 0) {
      console.warn("Save: Empty model.");
    }
    const numLayers = this.layers.length;

    const ensureLen = (arrName, dv = null) => {
      let currentArr = this[arrName];
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
      m_dw: ensureLen("m_dw"),
      v_dw: ensureLen("v_dw"),
      m_db: ensureLen("m_db"),
      v_db: ensureLen("v_db"),
      m_dgamma: ensureLen("m_dgamma"),
      v_dgamma: ensureLen("v_dgamma"),
      m_dbeta: ensureLen("m_dbeta"),
      v_dbeta: ensureLen("v_dbeta"),
      s_dw: ensureLen("s_dw"),
      s_db: ensureLen("s_db"),
      s_dgamma: ensureLen("s_dgamma"),
      s_dbeta: ensureLen("s_dbeta"),
    };

    const data = {
      weights: ensureLen("weights"),
      biases: ensureLen("biases"),
      gammas: ensureLen("gammas"),
      betas: ensureLen("betas"),
      layers: this.layers,
      details: this.details,
      usePositionalEncoding: this.usePositionalEncoding,
      optimizerState: optimizerState,
    };

    try {
      if (this.debug) console.log("Preparing data object:", data);
      const jsonStr = JSON.stringify(data);
      if (this.debug) console.log("Stringified JSON (length):", jsonStr.length);

      if (this.debug && data.weights[0] instanceof Float32Array) {
        console.log(
          "Sample stringified weight (should be object):",
          jsonStr.substring(0, 500).includes('"weights":[{"0":'),
        );
      }

      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${name}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      if (this.debug) console.log(`Model saved: ${name}.json`);
    } catch (e) {
      console.error("Save failed.", e);
      if (this.debug) console.error(" Error during stringify or download.");
    }
  }

  load(callback) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.style.display = "none";
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
          if (this.debug) console.log(" Reading file text...");
          const data = JSON.parse(text);
          if (!data.layers || !Array.isArray(data.layers))
            throw new Error("Invalid model: 'layers' missing.");
          if (this.debug)
            console.log(" Parsed data, layers found:", data.layers.length);

          this.reset();
          this.layers = data.layers;
          this.details = data.details || {};
          this.usePositionalEncoding = data.usePositionalEncoding || false;
          const numLayers = this.layers.length;

          const loadAndReconstruct = (arrName, sourceObj, expectedLen) => {
            let loadedArr = sourceObj?.[arrName] || [];
            if (!Array.isArray(loadedArr)) {
              console.warn(
                ` ${arrName} in loaded data is not an array, creating default.`,
              );
              loadedArr = [];
            }

            if (loadedArr.length !== expectedLen) {
              console.warn(
                ` ${arrName} length mismatch (expected ${expectedLen}, got ${loadedArr.length}). Adjusting...`,
              );
              const adjustedArr = Array(expectedLen).fill(null);
              for (let i = 0; i < Math.min(expectedLen, loadedArr.length); ++i)
                adjustedArr[i] = loadedArr[i];
              loadedArr = adjustedArr;
            }

            return loadedArr.map((item, index) => {
              if (
                item !== null &&
                typeof item === "object" &&
                item.hasOwnProperty("0")
              ) {
                const values = Object.values(item);

                const allNumbers = values.every(
                  (v) => typeof v === "number" && isFinite(v),
                );

                if (allNumbers) {
                  const reconstructed = new Float32Array(values);
                  if (this.debug)
                    console.log(
                      ` Reconstructed Float32Array for ${arrName}[${index}], Length: ${reconstructed.length}`,
                    );
                  return reconstructed;
                } else {
                  console.warn(
                    ` Object for ${arrName}[${index}] looks like Float32Array but check failed. Logging values:`,
                  );

                  let loggedCount = 0;
                  for (let k = 0; k < values.length && loggedCount < 5; k++) {
                    const v = values[k];
                    if (typeof v !== "number" || !isFinite(v)) {
                      console.warn(
                        `  - ${arrName}[${index}], Value[${k}]: Type=${typeof v}, Value=${v}`,
                      );
                      loggedCount++;
                    }
                  }
                  if (loggedCount === 0 && values.length > 0) {
                    console.warn(
                      `  - ${arrName}[${index}]: Check failed but couldn't find non-numeric/non-finite value? First value:`,
                      values[0],
                    );
                  }

                  return null;
                }
              } else if (item instanceof Float32Array) {
                if (this.debug)
                  console.log(
                    ` Item ${arrName}[${index}] is already Float32Array? Length: ${item.length}`,
                  );
                return item;
              } else if (item === null) {
                return null;
              } else {
                console.warn(
                  ` Unexpected item type for ${arrName}[${index}] (Type: ${typeof item}). Setting to null. Value:`,
                  item,
                );
                return null;
              }
            });
          };

          this.weights = loadAndReconstruct("weights", data, numLayers);
          this.biases = loadAndReconstruct("biases", data, numLayers);
          this.gammas = loadAndReconstruct("gammas", data, numLayers);
          this.betas = loadAndReconstruct("betas", data, numLayers);
          this.masks = Array(numLayers).fill(null);

          const optState = data.optimizerState || {};
          this.t = optState.t || 0;
          if (this.debug) console.log(" Loading optimizer state...");
          this.m_dw = loadAndReconstruct("m_dw", optState, numLayers);
          this.v_dw = loadAndReconstruct("v_dw", optState, numLayers);
          this.m_db = loadAndReconstruct("m_db", optState, numLayers);
          this.v_db = loadAndReconstruct("v_db", optState, numLayers);
          this.m_dgamma = loadAndReconstruct("m_dgamma", optState, numLayers);
          this.v_dgamma = loadAndReconstruct("v_dgamma", optState, numLayers);
          this.m_dbeta = loadAndReconstruct("m_dbeta", optState, numLayers);
          this.v_dbeta = loadAndReconstruct("v_dbeta", optState, numLayers);
          this.s_dw = loadAndReconstruct("s_dw", optState, numLayers);
          this.s_db = loadAndReconstruct("s_db", optState, numLayers);
          this.s_dgamma = loadAndReconstruct("s_dgamma", optState, numLayers);
          this.s_dbeta = loadAndReconstruct("s_dbeta", optState, numLayers);

          this.lastActivations = null;
          this.forwardCache = null;
          this.isTraining = false;
          if (callback) callback();
          if (this.debug)
            console.log(
              " Model loaded successfully. Stored parameters/states should be Float32Arrays or null.",
            );
          if (this.debug && this.weights.length > 0)
            console.log(
              " Sample loaded weight type:",
              this.weights[0] instanceof Float32Array
                ? "Float32Array"
                : typeof this.weights[0],
            );
        } catch (err) {
          console.error("Load failed:", err);
          alert(`Error loading model: ${err.message}`);
          if (this.debug)
            console.error(" Error during parsing or reconstruction.");
          if (callback) callback(err);
        } finally {
          cleanup();
        }
      };
      reader.onerror = (err) => {
        console.error("File read error:", err);
        alert("Error reading file.");
        cleanup();
        if (callback) callback(err);
      };
      reader.readAsText(file);
    };
    const cleanup = () => {
      input.removeEventListener("change", handleListener);
      document.body.removeChild(input);
    };
    input.addEventListener("change", handleListener);
    document.body.appendChild(input);
    input.click();
  }
}

