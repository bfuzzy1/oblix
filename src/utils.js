export const oblixUtils = {
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

      const isTypedArray =
        ArrayBuffer.isView(targetInfo) && !(targetInfo instanceof DataView);

      if (typeof targetInfo === "number" && Number.isInteger(targetInfo)) {
        targetIndex = targetInfo;
      } else if (
        (Array.isArray(targetInfo) || isTypedArray) &&
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
    const u1 = 1 - Math.random();
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    this._gaussian_spare = r * Math.sin(theta);
    return r * Math.cos(theta);
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

