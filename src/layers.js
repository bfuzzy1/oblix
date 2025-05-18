export const oblixLayerOps = {
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

    const randInts = new Uint32Array(N);
    const fill = context.randomFillFn;
    if (typeof fill === 'function') {
      fill(randInts);
    } else if (typeof globalThis.crypto !== 'undefined' &&
               typeof globalThis.crypto.getRandomValues === 'function') {
      globalThis.crypto.getRandomValues(randInts);
    } else {
      for (let i = 0; i < N; i++) {
        randInts[i] = (Math.random() * 0xffffffff) >>> 0;
      }
    }

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

