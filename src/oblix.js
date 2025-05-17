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

class oblix {
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

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
  const nn = new oblix(true);
  let lossHistory = [];
  const lossCanvas = document.getElementById("lossGraph");
  const networkCanvas = document.getElementById("networkGraph");
  const lossCtx = lossCanvas?.getContext("2d");
  const networkCtx = networkCanvas?.getContext("2d");
  const statsEl = document.getElementById("stats");
  const trainButton = document.getElementById("trainButton");
  const pauseButton = document.getElementById("pauseButton");
  const resumeButton = document.getElementById("resumeButton");
  const predictButton = document.getElementById("predictButton");
  const saveButton = document.getElementById("saveButton");
  const loadButton = document.getElementById("loadButton");
  const unloadButton = document.getElementById("unloadButton");
  const epochBar = document.getElementById("epochBar");
  const predictionResultEl = document.getElementById("predictionResult");
  const numHiddenLayersInput = document.getElementById("numHiddenLayers");
  const hiddenLayersConfigContainer =
    document.getElementById("hiddenLayersConfig");
  const optimizerSelect = document.getElementById("optimizer");
  const usePositionalEncodingCheckbox = document.getElementById(
    "usePositionalEncoding",
  );
  const lossFunctionSelect = document.getElementById("lossFunction");
  const l2LambdaInput = document.getElementById("l2Lambda");
  const decayRateGroup = document.getElementById("decayRateGroup");
  const decayRateInput = document.getElementById("decayRate");
  const gradientClipValueInput = document.getElementById("gradientClipValue");
  const trainingDataTextarea = document.getElementById("trainingData");
  const testDataTextarea = document.getElementById("testData");
  const epochsInput = document.getElementById("epochs");
  const learningRateInput = document.getElementById("learningRate");
  const batchSizeInput = document.getElementById("batchSize");
  const lrSchedulerSelect = document.getElementById("lrScheduler");
  const lrStepParamsDiv = document.getElementById("lrStepParams");
  const lrExpParamsDiv = document.getElementById("lrExpParams");
  const architectureTemplateSelect = document.getElementById(
    "architectureTemplateSelect",
  );
  const dataPatternSelect = document.getElementById("dataPattern");
  const numInputDimsInput = document.getElementById("numInputDims");
  const numOutputDimsInput = document.getElementById("numOutputDims");
  const numInputDimsGroup = numInputDimsInput.closest(".input-group");
  const numOutputDimsGroup = numOutputDimsInput.closest(".input-group");
  const inputDimsLabel = numInputDimsGroup?.querySelector("label");
  const outputDimsLabel = numOutputDimsGroup?.querySelector("label");

  function updateDataParamUI(selectedPattern) {
    let ignoreDims = false;
    let inputTitle = "Number of input features per sample";
    let outputTitle = "Number of output values per sample";

    switch (selectedPattern) {
      case "xor":
      case "circular":
        ignoreDims = true;
        inputTitle = "Input Dimensions (Fixed to 2 for this pattern)";
        outputTitle = "Output Dimensions (Fixed to 1 for this pattern)";
        if (numInputDimsInput) numInputDimsInput.value = 2;
        if (numOutputDimsInput) numOutputDimsInput.value = 1;
        break;
      case "blobs":
        ignoreDims = true;
        inputTitle = "Input Dimensions (Fixed to 2 for blobs)";

        outputTitle = "Number of Classes (Output Dim controls this)";
        if (numInputDimsInput) numInputDimsInput.value = 2;

        break;
      case "linear":
      case "random":
      default:
        ignoreDims = false;

        break;
    }

    if (numInputDimsInput) numInputDimsInput.disabled = ignoreDims;
    if (numOutputDimsInput) numOutputDimsInput.disabled = ignoreDims;

    if (numInputDimsGroup)
      numInputDimsGroup.classList.toggle("ignored-param", ignoreDims);
    if (numOutputDimsGroup)
      numOutputDimsGroup.classList.toggle("ignored-param", ignoreDims);

    if (inputDimsLabel) inputDimsLabel.title = inputTitle;
    if (outputDimsLabel) outputDimsLabel.title = outputTitle;
  }

  if (dataPatternSelect) {
    dataPatternSelect.addEventListener("change", (event) => {
      updateDataParamUI(event.target.value);
    });

    updateDataParamUI(dataPatternSelect.value);
  } else {
    console.error("Data Pattern select element not found.");
  }

  const architectureTemplates = {
    mlp: {
      numHidden: 2,
      layers: [
        { type: "dense", size: 16, activation: "relu", useBias: true },
        { type: "dense", size: 8, activation: "relu", useBias: true },
      ],
    },
    autoencoder: {
      numHidden: 3,
      layers: [
        { type: "dense", size: 16, activation: "relu", useBias: true },
        { type: "dense", size: 4, activation: "relu", useBias: true },
        { type: "dense", size: 16, activation: "relu", useBias: true },
      ],

      finalActivationHint: "sigmoid",
    },
    transformerEncoder: {
      numHidden: 8,
      layers: [
        { type: "layernorm", size: null },
        { type: "attention", numHeads: 3 },
        { type: "dense", size: 32, activation: "relu", useBias: true },
        { type: "dropout", rate: 0.1 },

        { type: "layernorm", size: null },
        { type: "attention", numHeads: 4 },
        { type: "dense", size: 32, activation: "relu", useBias: true },
        { type: "dropout", rate: 0.1 },
      ],

      finalActivationHint: "none",
    },
    residualAttention: {
      numHidden: 5,
      layers: [
        { type: "layernorm" },
        { type: "attention", numHeads: 3 },
        { type: "dense", size: 32, activation: "relu", useBias: true },
        { type: "dropout", rate: 0.1 },
        { type: "layernorm" },
      ],
      finalActivationHint: "none",
    },
  };

  try {
    function _wm(c) {
      if (!c) return;
      let d = Math.max(
          0,
          Math.floor((new Date() - new Date(2023, 0, 1)) / 864e5),
        ),
        r = 0x4f,
        g = (d >> 8) & 255,
        b = d & 255;
      c.fillStyle = `rgb(${r},${g},${b})`;
      c.fillRect(0, 0, 1, 1);
    }
    _wm(document.getElementById("watermarkCanvas")?.getContext("2d"));
  } catch (e) {
    console.error("Watermark err:", e);
  }

  function formatGeneratedDataToCSV(dataArray) {
    if (!Array.isArray(dataArray)) return [];
    return dataArray
      .map((sample) => {
        if (
          !sample ||
          !Array.isArray(sample.input) ||
          !Array.isArray(sample.output)
        ) {
          console.warn("Skipping invalid sample in formatGeneratedDataToCSV");
          return null;
        }

        return [...sample.input, ...sample.output]
          .map((v) =>
            typeof v === "number" && isFinite(v) ? v.toFixed(3) : "NaN",
          )
          .join(", ");
      })
      .filter((row) => row !== null);
  }

  function generateRandomData(
    numSamples,
    numInputs,
    numOutputs = 1,
    noiseLevel = 0.05,
  ) {
    if (numInputs <= 0 || numOutputs <= 0) return "";
    const data = [];
    for (let i = 0; i < numSamples; i++) {
      const input = [];
      for (let j = 0; j < numInputs; j++) input.push(Math.random());
      const output = [];
      for (let j = 0; j < numOutputs; j++) {
        const base = Math.sin(input[0] * Math.PI * 2) * 0.4 + 0.5;
        const noise = (Math.random() - 0.5) * 2 * noiseLevel;
        let final = Math.max(0.01, Math.min(0.99, base + noise));
        output.push(final);
      }
      data.push([...input, ...output].map((v) => v.toFixed(3)).join(", "));
    }
    return data.join("\n");
  }
  document.getElementById("generateDataBtn").addEventListener("click", () => {
    const numTrainInput =
      parseInt(document.getElementById("numTrainSamples").value) || 100;
    const numTestInput =
      parseInt(document.getElementById("numTestSamples").value) || 25;
    const numIn = parseInt(document.getElementById("numInputDims").value) || 3;
    const numOut =
      parseInt(document.getElementById("numOutputDims").value) || 1;
    const noise =
      parseFloat(document.getElementById("noiseLevel").value) || 0.05;
    const selectedPattern = document.getElementById("dataPattern").value;

    const safeNumTrain = Math.max(1, Math.min(5000, numTrainInput));
    const safeNumTest = Math.max(0, Math.min(1000, numTestInput));
    const safeNumIn = Math.max(1, Math.min(50, numIn));
    const safeNumOut = Math.max(1, Math.min(20, numOut));
    const safeNoise = Math.max(0, Math.min(1, noise));

    const totalSamples = safeNumTrain + safeNumTest;
    let generatedData = [];
    let patternDesc = "Default Random";
    let finalInputDims = safeNumIn;
    let finalOutputDims = safeNumOut;

    try {
      switch (selectedPattern) {
        case "xor":
          generatedData = oblixUtils.generateXORData(totalSamples, safeNoise);
          patternDesc = "XOR";
          finalInputDims = 2;
          finalOutputDims = 1;
          break;
        case "linear":
          generatedData = oblixUtils.generateLinearData(
            totalSamples,
            safeNoise,
            safeNumIn,
            safeNumOut,
          );
          patternDesc = "Linear";

          break;
        case "circular":
          generatedData = oblixUtils.generateCircularData(
            totalSamples,
            safeNoise,
          );
          patternDesc = "Circular";
          finalInputDims = 2;
          finalOutputDims = 1;
          break;
        case "blobs":
          const numClasses = Math.max(2, safeNumOut);
          generatedData = oblixUtils.generateGaussianBlobs(
            totalSamples,
            safeNoise,
            numClasses,
          );
          patternDesc = `${numClasses}-Class Gaussian Blobs`;
          finalInputDims = 2;
          finalOutputDims = 1;
          break;
        case "random":
        default:
          const randomCsvString = generateRandomData(
            totalSamples,
            safeNumIn,
            safeNumOut,
            safeNoise,
          );
          generatedData = randomCsvString.split("\n");
          patternDesc = "Default Random";
          break;
      }

      let csvRows = [];
      if (selectedPattern === "random") {
        csvRows = generatedData;
      } else {
        csvRows = formatGeneratedDataToCSV(generatedData);
      }

      if (csvRows.length < totalSamples) {
        console.warn(
          `Generated fewer rows (${csvRows.length}) than requested (${totalSamples}). Adjusting counts.`,
        );
      }
      const actualTotal = csvRows.length;
      const actualTrainCount = Math.min(safeNumTrain, actualTotal);
      const actualTestCount = Math.min(
        safeNumTest,
        actualTotal - actualTrainCount,
      );

      for (let i = actualTotal - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [csvRows[i], csvRows[j]] = [csvRows[j], csvRows[i]];
      }

      const trainCsvString = csvRows.slice(0, actualTrainCount).join("\n");
      const testCsvString = csvRows
        .slice(actualTrainCount, actualTrainCount + actualTestCount)
        .join("\n");

      trainingDataTextarea.value = trainCsvString;
      testDataTextarea.value = testCsvString;

      statsEl.innerHTML = `Generated ${actualTrainCount}/${actualTestCount} samples using <strong>${patternDesc}</strong> pattern (${finalInputDims} inputs, ${finalOutputDims} outputs, ${safeNoise.toFixed(2)} noise).`;

      try {
        const firstSample = parseCSV(trainCsvString)[0];
        if (firstSample && nn.layers && nn.layers.length > 0) {
          if (nn.layers[0]?.inputSize === firstSample.input.length) {
            nn.predict(firstSample.input);
          } else {
            console.warn(
              "Model input size doesn't match newly generated data. Skipping predict/redraw.",
            );
          }
        }
      } catch (e) {
        console.warn("Error during post-generation predict:", e);
      }

      drawNetwork();
    } catch (error) {
      console.error("Error during data generation:", error);
      statsEl.innerHTML = `<span class="error">Generation Error: ${error.message}</span>`;
    }
  });
  function parseCSV(csvString) {
    if (!csvString || typeof csvString !== "string") return [];
    return csvString
      .trim()
      .split("\n")
      .map((r) => r.trim())
      .filter((r) => r.length > 0)
      .map((r, idx) => {
        const v = r.split(",").map((v) => parseFloat(v.trim()));
        if (v.some(isNaN)) {
          console.warn(`R ${idx + 1} NaN`);
          return null;
        }
        if (v.length < 2) {
          console.warn(`R ${idx + 1} <2 vals`);
          return null;
        }
        const i = v.slice(0, -1),
          o = v.slice(-1);
        if (i.length === 0) {
          console.warn(`R ${idx + 1} no input`);
          return null;
        }
        return { input: i, output: o };
      })
      .filter((i) => i !== null);
  }
  function drawLossGraph() {
    if (!lossCtx || !lossCanvas) return;
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    if (lossHistory.length < 2) return;
    const trainL = lossHistory
      .map((h) => h.train)
      .filter((l) => l !== null && isFinite(l));
    const testL = lossHistory
      .map((h) => h.test)
      .filter((l) => l !== null && isFinite(l));
    let maxL = 0.1;
    if (trainL.length > 0) maxL = Math.max(maxL, ...trainL);
    if (testL.length > 0) maxL = Math.max(maxL, ...testL);
    maxL = Math.max(maxL, 0.1);
    const W = lossCanvas.width,
      H = lossCanvas.height,
      nPts = lossHistory.length,
      pH = H * 0.9,
      yOff = H * 0.05;
    const plot = (ctx, pts, c) => {
      ctx.strokeStyle = c;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let first = true;
      pts.forEach((p, i) => {
        if (p !== null && isFinite(p)) {
          const x = (i / Math.max(1, nPts - 1)) * W;
          const y = H - (p / maxL) * pH - yOff;
          if (first) {
            ctx.moveTo(x, y);
            first = false;
          } else {
            ctx.lineTo(x, y);
          }
        } else {
          first = true;
        }
      });
      ctx.stroke();
    };
    const trainC =
      getComputedStyle(document.body).getPropertyValue("--text")?.trim() ||
      "#fff";
    plot(
      lossCtx,
      lossHistory.map((h) => h.train),
      trainC,
    );
    plot(
      lossCtx,
      lossHistory.map((h) => h.test),
      "#87CEEB",
    );
  }

  function createLayerConfigUI(numLayers) {
    hiddenLayersConfigContainer.innerHTML = "";
    const activationTypes = [
      "tanh",
      "sigmoid",
      "relu",
      "leakyrelu",
      "gelu",
      "selu",
      "swish",
      "mish",
      "softmax",
      "none",
    ];
    const layerTypes = [
      "dense",
      "layernorm",
      "attention",
      "dropout",
      "softmax",
    ];
    if (numLayers === 0) {
      hiddenLayersConfigContainer.innerHTML =
        '<p class="layer-note">No hidden layers. Direct input-to-output connection (final layer added automatically).</p>';
      return;
    }
    for (let i = 0; i < numLayers; i++) {
      const layerGroup = document.createElement("div");
      layerGroup.className = "input-group settings-grid";
      const typeDiv = document.createElement("div");
      typeDiv.className = "input-group";
      const typeLabel = document.createElement("label");
      typeLabel.textContent = `Layer ${i + 1} Type:`;
      typeLabel.htmlFor = `layerType_${i}`;
      typeLabel.title = "Choose the operation for this layer...";
      const typeSelect = document.createElement("select");
      typeSelect.id = `layerType_${i}`;
      typeSelect.dataset.layerIndex = i;
      typeSelect.dataset.configType = "type";
      layerTypes.forEach((t) => {
        const o = document.createElement("option");
        o.value = t;
        o.textContent = t;
        if (t === "dense") o.selected = true;
        typeSelect.appendChild(o);
      });
      typeDiv.appendChild(typeLabel);
      typeDiv.appendChild(typeSelect);
      layerGroup.appendChild(typeDiv);
      const optionsDiv = document.createElement("div");
      optionsDiv.className = "layer-options-container";
      optionsDiv.dataset.layerIndex = i;
      layerGroup.appendChild(optionsDiv);

      typeSelect.addEventListener(
        "change",
        () => (architectureTemplateSelect.value = "custom"),
      );

      const updateOptionsUI = (idx, selType) => {
        const optsDiv = hiddenLayersConfigContainer.querySelector(
          `.layer-options-container[data-layer-index='${idx}']`,
        );
        if (!optsDiv) return;
        optsDiv.innerHTML = "";
        const createIn = (l, id, t, v, mn, st, cfg, nt = null, tt = null) => {
          const dv = document.createElement("div");
          dv.className = "input-group";
          const lb = document.createElement("label");
          lb.textContent = l;
          lb.htmlFor = id;
          if (tt) lb.title = tt;
          const ip = document.createElement("input");
          ip.type = t;
          ip.id = id;
          ip.value = v;
          if (mn !== null) ip.min = mn;
          if (st !== null) ip.step = st;
          ip.dataset.layerIndex = idx;
          ip.dataset.configType = cfg;
          dv.appendChild(lb);
          dv.appendChild(ip);
          if (nt) {
            const p = document.createElement("p");
            p.className = "layer-note";
            p.textContent = nt;
            dv.appendChild(p);
          }
          return dv;
        };
        const createSel = (l, id, opts, selV, cfg, tt = null) => {
          const dv = document.createElement("div");
          dv.className = "input-group";
          const lb = document.createElement("label");
          lb.textContent = l;
          lb.htmlFor = id;
          if (tt) lb.title = tt;
          const sel = document.createElement("select");
          sel.id = id;
          sel.dataset.layerIndex = idx;
          sel.dataset.configType = cfg;
          opts.forEach((o) => {
            const op = document.createElement("option");
            op.value = o;
            op.textContent = o;
            if (o === selV) op.selected = true;
            sel.appendChild(op);
          });
          dv.appendChild(lb);
          dv.appendChild(sel);
          return dv;
        };
        const createChk = (l, id, chkd, cfg, tt = null) => {
          const dv = document.createElement("div");
          dv.className = "input-group";
          const lb = document.createElement("label");
          const ip = document.createElement("input");
          ip.type = "checkbox";
          ip.id = id;
          ip.checked = chkd;
          ip.dataset.layerIndex = idx;
          ip.dataset.configType = cfg;
          const sp = document.createElement("span");
          sp.textContent = l;
          if (tt) sp.title = tt;
          lb.appendChild(ip);
          lb.appendChild(sp);
          dv.appendChild(lb);
          return dv;
        };
        const createNt = (txt) => {
          const p = document.createElement("p");
          p.className = "layer-note";
          p.textContent = txt;
          const dv = document.createElement("div");
          dv.style.gridColumn = "1/-1";
          dv.appendChild(p);
          return dv;
        };
        if (selType === "dense") {
          optsDiv.appendChild(
            createIn(
              "Nodes:",
              `layerNodes_${idx}`,
              "number",
              10,
              1,
              1,
              "size",
              null,
              "Num neurons.",
            ),
          );
          optsDiv.appendChild(
            createSel(
              "Activation:",
              `layerAct_${idx}`,
              activationTypes,
              "tanh",
              "activation",
              "Neuron output function.",
            ),
          );
          optsDiv.appendChild(
            createChk(
              "Use Bias:",
              `layerBias_${idx}`,
              true,
              "useBias",
              "Add learnable bias term?",
            ),
          );
        } else if (selType === "attention") {
          optsDiv.appendChild(
            createIn(
              "Num Heads:",
              `layerHeads_${idx}`,
              "number",
              2,
              1,
              1,
              "numHeads",
              null,
              "Parallel attention mechanisms. Input size must be divisible by heads.",
            ),
          );
          optsDiv.appendChild(
            createNt(
              "Input size must be divisible by Num Heads. Output size matches input.",
            ),
          );
        } else if (selType === "layernorm") {
          optsDiv.appendChild(
            createNt("Normalizes features across the feature dimension."),
          );
        } else if (selType === "dropout") {
          optsDiv.appendChild(
            createIn(
              "Dropout Rate:",
              `layerRate_${idx}`,
              "number",
              0.5,
              0,
              0.01,
              "rate",
              "Fraction of neurons to zero out during training (0 to <1). Helps prevent overfitting.",
              "Higher value means more dropout.",
            ),
          );
        } else if (selType === "softmax") {
          optsDiv.appendChild(
            createNt(
              "Outputs probabilities summing to 1. For multi-class classification.",
            ),
          );
        }
      };
      typeSelect.addEventListener("change", (event) =>
        updateOptionsUI(i, event.target.value),
      );
      hiddenLayersConfigContainer.appendChild(layerGroup);
      updateOptionsUI(i, typeSelect.value);
    }
  }
  numHiddenLayersInput.addEventListener("change", (event) => {
    const numLayers = Math.max(0, parseInt(event.target.value) || 0);
    event.target.value = numLayers;
    createLayerConfigUI(numLayers);

    architectureTemplateSelect.value = "custom";
  });
  createLayerConfigUI(parseInt(numHiddenLayersInput.value));
  optimizerSelect.addEventListener("change", () => {
    decayRateGroup.style.display =
      optimizerSelect.value === "rmsprop" ? "block" : "none";
  });
  decayRateGroup.style.display =
    optimizerSelect.value === "rmsprop" ? "block" : "none";

  lrSchedulerSelect.addEventListener("change", () => {
    const selectedSchedule = lrSchedulerSelect.value;
    lrStepParamsDiv.style.display =
      selectedSchedule === "step" ? "grid" : "none";
    lrExpParamsDiv.style.display =
      selectedSchedule === "exponential" ? "grid" : "none";
  });
  lrSchedulerSelect.dispatchEvent(new Event("change"));

  architectureTemplateSelect.addEventListener("change", (event) => {
    const templateKey = event.target.value;
    if (templateKey === "custom") {
      return;
    }

    const template = architectureTemplates[templateKey];
    if (!template) {
      console.warn("Selected template not found:", templateKey);
      return;
    }

    statsEl.innerHTML = `Applying '${templateKey}' template...`;
    numHiddenLayersInput.value = template.numHidden;
    numHiddenLayersInput.dispatchEvent(new Event("change"));

    setTimeout(() => {
      try {
        template.layers.forEach((layerConfig, i) => {
          const setV = (s, v) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e && v !== undefined) e.value = v;
          };
          const setC = (s, c) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e && c !== undefined) e.checked = c;
          };
          const triggerChange = (s) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e) e.dispatchEvent(new Event("change", { bubbles: true }));
          };

          setV(
            `select[data-layer-index="${i}"][data-config-type="type"]`,
            layerConfig.type || "dense",
          );
          triggerChange(
            `select[data-layer-index="${i}"][data-config-type="type"]`,
          );

          switch (layerConfig.type) {
            case "dense":
              setV(
                `input[data-layer-index="${i}"][data-config-type="size"]`,
                layerConfig.size,
              );
              setV(
                `select[data-layer-index="${i}"][data-config-type="activation"]`,
                layerConfig.activation,
              );
              setC(
                `input[data-layer-index="${i}"][data-config-type="useBias"]`,
                layerConfig.useBias,
              );
              break;
            case "attention":
              setV(
                `input[data-layer-index="${i}"][data-config-type="numHeads"]`,
                layerConfig.numHeads,
              );
              break;
            case "dropout":
              setV(
                `input[data-layer-index="${i}"][data-config-type="rate"]`,
                layerConfig.rate,
              );
              break;
          }
        });

        statsEl.innerHTML = `Applied <span style="color: #87CEEB; font-weight: bold;">${templateKey}</span> template. Ready.`;
      } catch (error) {
        console.error("Error applying template UI:", error);
        statsEl.innerHTML = `<span class="error">Error applying template: ${error.message}</span>`;
        architectureTemplateSelect.value = "custom";
      }
    }, 0);
  });

  trainButton.addEventListener("click", async () => {
    statsEl.innerHTML = "Starting training...";
    trainButton.disabled = true;
    trainButton.textContent = "Training...";
    pauseButton.disabled = false;
    resumeButton.disabled = true;
    predictButton.disabled = true;
    saveButton.disabled = true;
    loadButton.disabled = true;
    unloadButton.disabled = true;
    epochBar.style.width = "0%";
    lossHistory = [];
    drawLossGraph();
    try {
      const weightInitMethod = document.getElementById("weightInit").value;

      const trainingData = parseCSV(trainingDataTextarea.value);
      const testData = parseCSV(testDataTextarea.value);
      if (trainingData.length === 0)
        throw new Error("Training data empty/invalid.");
      if (!trainingData[0]?.input || !trainingData[0]?.output)
        throw new Error("Cannot get input/output size from data.");
      nn.reset();
      const numHidden = parseInt(numHiddenLayersInput.value);
      const layerCfgs = [];
      const numIns = trainingData[0].input.length;
      let curInSize = numIns;
      for (let i = 0; i < numHidden; i++) {
        const getV = (s) => hiddenLayersConfigContainer.querySelector(s)?.value;
        const getC = (s) =>
          hiddenLayersConfigContainer.querySelector(s)?.checked;
        const lType =
          getV(`select[data-layer-index="${i}"][data-config-type="type"]`) ||
          "dense";

        let cfg = {
          type: lType,
          inputSize: curInSize,
          weightInit: weightInitMethod,
        };
        switch (lType) {
          case "dense":
            cfg.outputSize = parseInt(
              getV(`input[data-layer-index="${i}"][data-config-type="size"]`) ||
                1,
            );
            if (cfg.outputSize <= 0)
              throw new Error(`L${i + 1} Dense: Invalid nodes.`);
            cfg.activation =
              getV(
                `select[data-layer-index="${i}"][data-config-type="activation"]`,
              ) || "tanh";
            cfg.useBias =
              getC(
                `input[data-layer-index="${i}"][data-config-type="useBias"]`,
              ) ?? true;
            break;
          case "attention":
            cfg.numHeads = parseInt(
              getV(
                `input[data-layer-index="${i}"][data-config-type="numHeads"]`,
              ) || 1,
            );
            if (cfg.numHeads <= 0)
              throw new Error(`L${i + 1} Attn: Invalid heads.`);
            if (curInSize % cfg.numHeads !== 0)
              throw new Error(
                `L${i + 1} Attn: Input ${curInSize} not divisible by ${cfg.numHeads} heads.`,
              );
            cfg.outputSize = curInSize;
            break;
          case "dropout":
            cfg.rate = parseFloat(
              getV(`input[data-layer-index="${i}"][data-config-type="rate"]`) ||
                0,
            );
            if (cfg.rate < 0 || cfg.rate >= 1)
              throw new Error(`L${i + 1} Dropout: Invalid rate.`);
            cfg.outputSize = curInSize;
            break;
          case "layernorm":
          case "softmax":
            cfg.outputSize = curInSize;
            break;
          default:
            throw new Error(`L${i + 1}: Unknown type "${lType}".`);
        }
        if (!cfg.outputSize) throw new Error(`L${i + 1}: No output size.`);
        layerCfgs.push(cfg);
        curInSize = cfg.outputSize;
      }

      const isAutoencoder = architectureTemplateSelect.value === "autoencoder";
      const numOuts = isAutoencoder ? numIns : trainingData[0].output.length;
      if (numOuts <= 0) throw new Error("Zero output cols defined.");

      const selectedLoss = lossFunctionSelect.value;

      let finalAct = "none";

      if (selectedLoss === "crossentropy") {
        if (isAutoencoder) {
          finalAct =
            architectureTemplates["autoencoder"]?.finalActivationHint ||
            "sigmoid";
          console.log(
            `Autoencoder with Cross-Entropy? Using template hint or sigmoid: ${finalAct}`,
          );
        } else if (numOuts > 1) {
          finalAct = "softmax";
        } else if (numOuts === 1) {
          finalAct = "sigmoid";
        }
      } else if (selectedLoss === "mse") {
        if (isAutoencoder) {
          finalAct =
            architectureTemplates["autoencoder"]?.finalActivationHint ||
            "sigmoid";
          console.log(
            `Autoencoder with MSE, using template hint or sigmoid: ${finalAct}`,
          );
        } else {
          finalAct = "none";
          console.log("MSE loss selected, using linear final activation: none");
        }
      }

      console.log(
        `Adding final dense: ${curInSize}->${numOuts} (Act:${finalAct})`,
      );
      layerCfgs.push({
        type: "dense",
        inputSize: curInSize,
        outputSize: numOuts,
        activation: finalAct,
        useBias: true,
      });

      layerCfgs.forEach((cfg, i) => {
        try {
          nn.layer(cfg);
        } catch (e) {
          throw new Error(`Cfg L${i + 1}(${cfg.type}): ${e.message}`);
        }
      });
      if (nn.layers.length === 0) throw new Error("Zero layers configured.");
      if (nn.debug) console.log("Net structure:", nn.layers);
      const opts = {
        epochs: parseInt(epochsInput.value) || 50,
        learningRate: parseFloat(learningRateInput.value) || 0.01,
        batchSize: parseInt(batchSizeInput.value) || 8,
        testSet: testData.length > 0 ? testData : null,
        optimizer: optimizerSelect.value,
        lossFunction: lossFunctionSelect.value,
        l2Lambda: parseFloat(l2LambdaInput.value) || 0,
        decayRate: parseFloat(decayRateInput.value) || 0.9,
        gradientClipValue: parseFloat(gradientClipValueInput.value) || 0,
        usePositionalEncoding: usePositionalEncodingCheckbox.checked,
        lrSchedule: lrSchedulerSelect.value,
        lrStepDecayFactor:
          parseFloat(document.getElementById("lrStepDecayFactor").value) || 0.1,
        lrStepDecaySize:
          parseInt(document.getElementById("lrStepDecaySize").value) || 10,
        lrExpDecayRate:
          parseFloat(document.getElementById("lrExpDecayRate").value) || 0.95,
        callback: async (
          ep,
          trL,
          tstL,
          metricName,
          metricVal,
          lastForwardCache,
        ) => {
          lossHistory.push({ train: trL, test: tstL });
          drawLossGraph();
          epochBar.style.width = `${(ep / opts.epochs) * 100}%`;

          const currentLR = nn.getCurrentLearningRate(
            ep - 1,
            opts.learningRate,
            opts,
          );
          const lrString =
            opts.lrSchedule !== "none"
              ? ` | LR: ${currentLR.toExponential(2)}`
              : "";

          let statusText = `Ep:${ep}/${opts.epochs} | Loss:${trL.toFixed(6)}`;
          if (tstL !== null) {
            statusText += ` | Val Loss:${tstL.toFixed(6)}`;
          }

          if (metricName && metricVal !== null && !isNaN(metricVal)) {
            statusText += ` | Val ${metricName}:${metricVal.toFixed(4)}`;
          }
          statusText += lrString;
          statsEl.innerHTML = statusText;

          if (
            lastForwardCache &&
            lastForwardCache.activations &&
            lastForwardCache.activations.length > 0
          ) {
            nn.lastActivations = lastForwardCache.activations;

            if (!window._drawNetworkScheduled) {
              window._drawNetworkScheduled = true;
              requestAnimationFrame(() => {
                try {
                  drawNetwork();
                } catch (drawErr) {
                  console.warn("Error during dynamic drawNetwork:", drawErr);
                } finally {
                  window._drawNetworkScheduled = false;
                }
              });
            }
          }
        },
      };
      statsEl.innerHTML =
        `Training (${opts.optimizer}, ${opts.lossFunction}` +
        (opts.lrSchedule !== "none" ? `, ${opts.lrSchedule} LR` : "") +
        `)...`;

      window._drawNetworkScheduled = false;

      const summary = await nn.train(trainingData, opts);
      const totalParams = nn.getTotalParameters();
      statsEl.innerHTML =
        `<strong>Done!</strong> Loss:${summary.trainLoss.toFixed(6)}` +
        (summary.testLoss !== null
          ? `, Val:${summary.testLoss.toFixed(6)}`
          : "") +
        ` | Params:${totalParams.toLocaleString()}`;
      console.log("Final Summary:", summary);
      if (trainingData.length > 0) {
        try {
          nn.predict(trainingData[0].input);
          drawNetwork();
        } catch (e) {}
      }
    } catch (error) {
      console.error("Train err:", error);
      statsEl.innerHTML = `<span class="error">Error: ${error.message}</span>`;
    } finally {
      trainButton.disabled = false;
      trainButton.textContent = "Train Model";
      predictButton.disabled = false;
      saveButton.disabled = false;
      loadButton.disabled = false;
      unloadButton.disabled = false;
      pauseButton.disabled = true;
      resumeButton.disabled = true;
    }
  });

  pauseButton.addEventListener("click", () => {
    nn.pauseTraining();
    pauseButton.disabled = true;
    resumeButton.disabled = false;
    statsEl.innerHTML = "Training paused";
  });

  resumeButton.addEventListener("click", () => {
    nn.resumeTraining();
    pauseButton.disabled = false;
    resumeButton.disabled = true;
  });

  function drawNetwork() {
    if (!networkCtx || !networkCanvas) return;
    const networkContainer = networkCanvas.parentElement;
    if (!networkContainer) return;
    const containerWidth = networkContainer.clientWidth;
    const containerHeight = networkContainer.clientHeight;
    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
    const hasModel =
      nn.lastActivations &&
      nn.lastActivations.length > 0 &&
      nn.layers &&
      nn.layers.length > 0;

    if (!hasModel) {
      networkCtx.fillStyle = "#555";
      networkCtx.font = "10px monospace";
      networkCtx.textAlign = "center";
      networkCtx.textBaseline = "middle";

      if (networkCanvas.width !== containerWidth)
        networkCanvas.width = containerWidth;
      if (networkCanvas.height !== containerHeight)
        networkCanvas.height = containerHeight;
      networkCtx.fillText(
        "Train/Predict to visualize",
        containerWidth / 2,
        containerHeight / 2,
      );
      return;
    }

    const pad = 35,
      maxNds = 20,
      nRBase = 2,
      nRScale = 3,
      cBOp = 0.02,
      cMOp = 0.85,
      cWScale = 2,
      ellOff = 10,
      lblOff = 20,
      lblFnt = "10px monospace",
      lblClr = "#aaa";
    const nVizLyrs = nn.lastActivations.length;
    const baseSpacing = 150;
    const minLayerSpacing = Math.max(
      120,
      Math.min(baseSpacing, containerWidth / (nVizLyrs > 1 ? nVizLyrs : 1)),
    );

    const requiredWidth =
      nVizLyrs <= 1
        ? containerWidth
        : pad * 2 + (nVizLyrs - 1) * minLayerSpacing;

    const canvasDrawWidth = Math.max(containerWidth, requiredWidth * 1.2);

    networkCanvas.width = canvasDrawWidth;
    networkCanvas.height = containerHeight;

    const drawAreaWidth = canvasDrawWidth - pad * 2;
    const drawAreaHeight = containerHeight - pad * 2;
    const layerXs = Array.from(
      { length: nVizLyrs },
      (_, i) =>
        pad +
        (nVizLyrs === 1
          ? drawAreaWidth / 2
          : (drawAreaWidth * i) / (nVizLyrs - 1)),
    );

    const layerPos = [];
    nn.lastActivations.forEach((act, lIdx) => {
      if (!act || typeof act.length !== "number") {
        if (nn.debug)
          console.warn(
            `drawNetwork L${lIdx}: Activation data is not array-like.`,
            act,
          );
        layerPos.push([]);
        return;
      }

      const lNodes = [];
      const nNodes = act.length;
      const dNodes = Math.min(nNodes, maxNds);
      const lX = layerXs[lIdx];
      for (let j = 0; j < dNodes; j++) {
        const origIdx =
          nNodes <= maxNds ? j : Math.floor((j * nNodes) / dNodes);
        const nodeVal = act[origIdx];
        const nY =
          pad +
          (dNodes === 1
            ? drawAreaHeight / 2
            : (drawAreaHeight * j) / (dNodes - 1));

        lNodes.push({
          x: lX,
          y: nY,
          value: typeof nodeVal === "number" && isFinite(nodeVal) ? nodeVal : 0,
        });
      }
      if (nNodes > maxNds) {
        lNodes.push({
          x: lX,
          y: pad + drawAreaHeight + ellOff,
          value: 0,
          isEllipsis: true,
          originalCount: nNodes,
        });
      }
      layerPos.push(lNodes);
    });

    networkCtx.lineWidth = 1;
    for (let i = 0; i < nVizLyrs - 1; i++) {
      const curNodes = layerPos[i].filter((n) => !n.isEllipsis);
      const nextNodes = layerPos[i + 1].filter((n) => !n.isEllipsis);
      const cfg = nn.layers[i];
      if (!cfg) continue;

      const isDenseW =
        cfg.type === "dense" && nn.weights?.[i] instanceof Float32Array;
      const w = isDenseW ? nn.weights[i] : null;
      const inputSizeForLayer = cfg.inputSize;

      for (let j = 0; j < curNodes.length; j++) {
        for (let k = 0; k < nextNodes.length; k++) {
          let op = 0.1,
            col = "100,100,100",
            lw = 0.5;
          let lineDash = [];
          let weight = null;

          if (
            isDenseW &&
            w &&
            typeof inputSizeForLayer === "number" &&
            inputSizeForLayer > 0
          ) {
            const weightIndex = k * inputSizeForLayer + j;
            if (weightIndex >= 0 && weightIndex < w.length) {
              weight = w[weightIndex];
            } else {
              if (nn.debug)
                console.warn(
                  `drawNetwork L${i} Dense: Invalid weight index ${weightIndex} (nextIdx=${k}, currIdx=${j}, inSize=${inputSizeForLayer}, wLen=${w.length})`,
                );
              weight = null;
            }
          }

          if (
            isDenseW &&
            weight !== null &&
            typeof weight === "number" &&
            typeof curNodes[j].value === "number"
          ) {
            const wMag = Math.tanh(Math.abs(weight));
            const aMag = Math.tanh(Math.abs(curNodes[j].value));
            const combSig = wMag * 0.8 + aMag * 0.2;
            op = Math.min(Math.max(cBOp + combSig * (cMOp - cBOp), cBOp), cMOp);
            col = weight >= 0 ? "255,255,255" : "180,180,255";
            lw = Math.min(Math.max(0.5, op * cWScale), 2);
            lineDash = [];
          } else if (cfg.type === "attention") {
            op = 0.5;
            lw = 1.2;
            col = "0,200,200";
            lineDash = [];
          } else if (cfg.type === "layernorm") {
            op = 0.4;
            lw = 0.8;
            col = "255,255,0";
            lineDash = [2, 2];
          } else if (cfg.type === "dropout") {
            op = 0.2;
            lw = 0.6;
            col = "255,165,0";
            lineDash = [3, 3];
          } else if (cfg.type === "softmax") {
            op = 0.4;
            lw = 1.0;
            col = "200,0,200";
            lineDash = [];
          } else if (!isDenseW) {
            op = 0.1;
            lw = 0.5;
            col = "100,100,100";
            lineDash = [];
          } else {
            op = 0.05;
            lw = 0.3;
            col = "80,80,80";
            lineDash = [1, 2];
          }

          networkCtx.strokeStyle = `rgba(${col},${op})`;
          networkCtx.lineWidth = lw;
          networkCtx.setLineDash(lineDash || []);
          networkCtx.beginPath();
          networkCtx.moveTo(curNodes[j].x, curNodes[j].y);
          networkCtx.lineTo(nextNodes[k].x, nextNodes[k].y);
          networkCtx.stroke();
          networkCtx.setLineDash([]);
        }
      }
    }

    networkCtx.textAlign = "center";

    if (layerXs.length > 2) {
      const firstHiddenLayerX = layerXs[1];
      const lastHiddenLayerX = layerXs[layerXs.length - 2];
      const centerX = (firstHiddenLayerX + lastHiddenLayerX) / 2;

      networkCtx.fillStyle = lblClr;
      networkCtx.font = lblFnt;
      networkCtx.textBaseline = "bottom";
      networkCtx.fillText("Layers", centerX, pad - lblOff / 2);
    }

    layerPos.forEach((lNodes, lIdx) => {
      networkCtx.fillStyle = lblClr;
      networkCtx.font = lblFnt;
      networkCtx.textBaseline = "bottom";
      if (lIdx === 0) {
        networkCtx.fillText("Input", layerXs[lIdx], pad - lblOff / 2);
      } else if (lIdx === layerPos.length - 1) {
        networkCtx.fillText("Output", layerXs[lIdx], pad - lblOff / 2);
      }

      lNodes.forEach((n) => {
        if (n.isEllipsis) {
          networkCtx.fillStyle = "#777";
          networkCtx.font = "10px monospace";
          networkCtx.textBaseline = "top";
          networkCtx.fillText(`(${n.originalCount} nodes)`, n.x, n.y);
        } else {
          const actStr = Math.tanh(Math.abs(n.value));
          const r = nRBase + actStr * nRScale;
          const op = 0.3 + actStr * 0.7;
          const col = n.value >= 0 ? "255,255,255" : "200,200,255";
          networkCtx.fillStyle = `rgba(${col},${op})`;
          networkCtx.strokeStyle = "rgba(255,255,255,0.6)";
          networkCtx.lineWidth = 1;
          networkCtx.beginPath();
          networkCtx.arc(n.x, n.y, r, 0, Math.PI * 2);
          networkCtx.fill();
          networkCtx.stroke();
        }
      });
    });
  }
  function resizeCanvases() {
    const lossContainer = lossCanvas?.parentElement;

    const networkContainer = document.getElementById("network-viz-container");

    if (lossContainer?.clientWidth > 0 && lossCanvas) {
      lossCanvas.width = lossContainer.clientWidth;
      lossCanvas.height = lossContainer.clientHeight;
      drawLossGraph();
    }

    if (networkContainer?.clientWidth > 0 && networkCanvas) {
      drawNetwork();
    }
  }
  window.addEventListener("resize", resizeCanvases);
  setTimeout(resizeCanvases, 150);

  saveButton.addEventListener("click", () => {
    if (!nn.layers || nn.layers.length === 0) {
      statsEl.innerHTML = '<span class="error">No model to save.</span>';
      return;
    }

    const userProvidedName = window.prompt(
      "Enter filename base (e.g., my_model):",
      "oblix_model",
    );

    if (userProvidedName === null) {
      statsEl.innerHTML = "Save cancelled.";
      return;
    }

    let filenameBase;
    if (userProvidedName.trim() === "") {
      const now = new Date();
      const year = now.getUTCFullYear();
      const month = (now.getUTCMonth() + 1).toString().padStart(2, "0");
      const day = now.getUTCDate().toString().padStart(2, "0");
      const hours = now.getUTCHours().toString().padStart(2, "0");
      const minutes = now.getUTCMinutes().toString().padStart(2, "0");
      const seconds = now.getUTCSeconds().toString().padStart(2, "0");
      filenameBase = `oblix_${year}${month}${day}_${hours}${minutes}${seconds}Z`;
      statsEl.innerHTML = `Using default filename: ${filenameBase}.json`;
    } else {
      filenameBase = userProvidedName.trim();
    }

    nn.save(filenameBase);
    statsEl.innerHTML = `Model saved as ${filenameBase}.json`;
  });

  loadButton.addEventListener("click", () => {
    statsEl.innerHTML = "Loading...";
    nn.load((error) => {
      if (error) {
        statsEl.innerHTML = `<span class="error">Load failed: ${error.message}</span>`;
        return;
      }
      const params = nn.getTotalParameters();
      statsEl.innerHTML = `<strong>Model loaded!</strong> Params: ${params.toLocaleString()}`;
      try {
        const d = nn.details?.training;
        const l = nn.layers || [];
        usePositionalEncodingCheckbox.checked =
          nn.usePositionalEncoding || false;
        if (d) {
          epochsInput.value = d.epochs || 50;
          learningRateInput.value = d.learningRate || 0.01;
          batchSizeInput.value = d.batchSize || 8;
          optimizerSelect.value = d.optimizer || "adam";
          lossFunctionSelect.value = d.lossFunction || "mse";
          l2LambdaInput.value = d.l2Lambda || 0;
          decayRateInput.value = d.decayRate || 0.9;
          gradientClipValueInput.value = d.gradientClipValue || 0;
          optimizerSelect.dispatchEvent(new Event("change"));
        }
        const numHid = Math.max(0, l.length - 1);
        numHiddenLayersInput.value = numHid;
        createLayerConfigUI(numHid);
        l.slice(0, numHid).forEach((layer, i) => {
          const setV = (s, v) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e && v !== undefined) e.value = v;
          };
          const setC = (s, c) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e && c !== undefined) e.checked = c;
          };
          setV(
            `select[data-layer-index="${i}"][data-config-type="type"]`,
            layer.type || "dense",
          );
          const ts = hiddenLayersConfigContainer.querySelector(
            `select[data-layer-index="${i}"][data-config-type="type"]`,
          );
          if (ts) ts.dispatchEvent(new Event("change", { bubbles: true }));
          switch (layer.type) {
            case "dense":
              setV(
                `input[data-layer-index="${i}"][data-config-type="size"]`,
                layer.outputSize,
              );
              setV(
                `select[data-layer-index="${i}"][data-config-type="activation"]`,
                layer.activation,
              );
              setC(
                `input[data-layer-index="${i}"][data-config-type="useBias"]`,
                layer.useBias,
              );
              break;
            case "attention":
              setV(
                `input[data-layer-index="${i}"][data-config-type="numHeads"]`,
                layer.numHeads,
              );
              break;
            case "dropout":
              setV(
                `input[data-layer-index="${i}"][data-config-type="rate"]`,
                layer.rate,
              );
              break;
          }
        });

        architectureTemplateSelect.value = "custom";
        lossHistory = [];
        drawLossGraph();
        predictionResultEl.innerHTML = "Result: -";
        try {
          const sample = parseCSV(trainingDataTextarea.value)[0];
          if (sample) nn.predict(sample.input);
        } catch (e) {}
        drawNetwork();
      } catch (uiError) {
        console.error("UI update err after load:", uiError);
        statsEl.innerHTML += ` <span class="error">(UI update failed)</span>`;
      }
    });
  });
  predictButton.addEventListener("click", () => {
    predictionResultEl.innerHTML = `Predicting...`;
    try {
      const inputStr = document.getElementById("predictionInput").value;
      if (!inputStr) throw new Error("Input empty.");
      const input = inputStr.split(",").map((s) => parseFloat(s.trim()));
      if (input.some(isNaN)) throw new Error("Invalid input.");
      if (!nn.layers || nn.layers.length === 0)
        throw new Error("Network not init.");
      const expectSz = nn.layers[0]?.inputSize;
      if (expectSz === undefined) throw new Error("Cannot get input size.");
      if (input.length !== expectSz)
        throw new Error(
          `Input size mismatch: Exp ${expectSz}, got ${input.length}.`,
        );
      const pred = nn.predict(input);
      if (pred === null) throw new Error("Prediction failed.");
      const predStr = pred.map((p) => p.toFixed(5)).join(", ");
      predictionResultEl.innerHTML = `Result: [${predStr}]`;
      drawNetwork();
    } catch (error) {
      console.error("Predict error:", error);
      predictionResultEl.innerHTML = `<span class="error">Error: ${error.message}</span>`;
    }
  });
  unloadButton.addEventListener("click", () => {
    console.log("Unload clicked.");
    try {
      nn.reset();
      lossHistory = [];
      drawLossGraph();
      drawNetwork();
      epochBar.style.width = "0%";
      statsEl.innerHTML = "Status: Model unloaded.";
      predictionResultEl.innerHTML = "Result: -";
      const defaultLayers = 2;
      numHiddenLayersInput.value = defaultLayers;
      createLayerConfigUI(defaultLayers);

      architectureTemplateSelect.value = "custom";
      console.log("Model & UI reset.");
    } catch (error) {
      console.error("Unload error:", error);
      statsEl.innerHTML = `<span class="error">Unload error: ${error.message}</span>`;
    }
  });
});
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = oblix;
}
