export const oblixOptimizers = {
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
        w instanceof Float32Array &&
        w.length > 0;
      const b = context.biases[i];
      const reqB =
        cfg.type === "dense" && cfg.useBias && b instanceof Float32Array && b.length > 0;
      const g = context.gammas[i];
      const beta = context.betas[i];
      const reqLN =
        cfg.type === "layernorm" &&
        g instanceof Float32Array &&
        beta instanceof Float32Array &&
        g.length === beta.length &&
        g.length > 0;

      if (
        optimizer === "adam" ||
        optimizer === "rmsprop" ||
        optimizer === "adamw"
      ) {
        if (reqW) {
          try {
            const size = w.length;
            if (optimizer === "adam" || optimizer === "adamw") {
              context.m_dw[i] = new Float32Array(size).fill(0);
              context.v_dw[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === "rmsprop") {
              context.s_dw[i] = new Float32Array(size).fill(0);
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
            const size = b.length;
            if (optimizer === "adam" || optimizer === "adamw") {
              context.m_db[i] = new Float32Array(size).fill(0);
              context.v_db[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === "rmsprop") {
              context.s_db[i] = new Float32Array(size).fill(0);
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
            const size = g.length;
            if (optimizer === "adam" || optimizer === "adamw") {
              context.m_dgamma[i] = new Float32Array(size).fill(0);
              context.v_dgamma[i] = new Float32Array(size).fill(0);
              context.m_dbeta[i] = new Float32Array(size).fill(0);
              context.v_dbeta[i] = new Float32Array(size).fill(0);
            }
            if (optimizer === "rmsprop") {
              context.s_dgamma[i] = new Float32Array(size).fill(0);
              context.s_dbeta[i] = new Float32Array(size).fill(0);
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

