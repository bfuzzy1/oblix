import assert from 'assert';
import { oblixOptimizers } from '../src/optimizers.js';

export async function run() {
  // Test initializeState with simple context
  const ctxInit = {
    debug: false,
    layers: [
      { type: 'dense', inputSize: 1, outputSize: 1, useBias: true },
      { type: 'layernorm', inputSize: 1 },
    ],
    weights: [[ [0.2] ] , null],
    biases: [[0.1], null],
    gammas: [null, [1, 1]],
    betas: [null, [0, 0]],
  };
  oblixOptimizers.initializeState(ctxInit, 'adam');
  assert.deepStrictEqual(ctxInit.m_dw[0], [[0]]);
  assert.deepStrictEqual(ctxInit.m_db[0], [0]);
  assert.deepStrictEqual(ctxInit.m_dgamma[1], [0, 0]);
  assert.deepStrictEqual(ctxInit.m_dbeta[1], [0, 0]);

  function createCtx() {
    return {
      debug: false,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
      t: 0,
      layers: [{ type: 'dense', inputSize: 1, outputSize: 1, useBias: true }],
      weights: [new Float32Array([0.5])],
      biases: [new Float32Array([0.2])],
      gammas: [null],
      betas: [null],
      m_dw: [null],
      v_dw: [null],
      m_db: [null],
      v_db: [null],
      m_dgamma: [null],
      v_dgamma: [null],
      m_dbeta: [null],
      v_dbeta: [null],
      s_dw: [null],
      s_db: [null],
      s_dgamma: [null],
      s_dbeta: [null],
    };
  }

  // SGD update
  const ctxSgd = createCtx();
  oblixOptimizers.updateParameters(
    ctxSgd,
    [new Float32Array([0.1])],
    [new Float32Array([0.05])],
    [],
    [],
    {
      learningRate: 0.1,
      initialLearningRate: 0.1,
      optimizer: 'sgd',
      batchSize: 1,
      l2Lambda: 0,
      gradientClipValue: 0,
      decayRate: 0.9,
    },
  );
  assert.ok(Math.abs(ctxSgd.weights[0][0] - 0.49) < 1e-6);
  assert.ok(Math.abs(ctxSgd.biases[0][0] - 0.195) < 1e-6);
  assert.strictEqual(ctxSgd.t, 1);

  // RMSprop update
  const ctxRms = createCtx();
  ctxRms.s_dw[0] = new Float32Array([0]);
  ctxRms.s_db[0] = new Float32Array([0]);
  oblixOptimizers.updateParameters(
    ctxRms,
    [new Float32Array([0.1])],
    [new Float32Array([0.05])],
    [],
    [],
    {
      learningRate: 0.1,
      initialLearningRate: 0.1,
      optimizer: 'rmsprop',
      batchSize: 1,
      l2Lambda: 0,
      gradientClipValue: 0,
      decayRate: 0.9,
    },
  );
  const sExpRW = 0.9 * 0 + 0.1 * 0.1 * 0.1;
  const updRW = (0.1 * 0.1) / (Math.sqrt(sExpRW) + 1e-8);
  const sExpRB = 0.9 * 0 + 0.1 * 0.05 * 0.05;
  const updRB = (0.1 * 0.05) / (Math.sqrt(sExpRB) + 1e-8);
  assert.ok(Math.abs(ctxRms.s_dw[0][0] - sExpRW) < 1e-6);
  assert.ok(Math.abs(ctxRms.weights[0][0] - (0.5 - updRW)) < 1e-6);
  assert.ok(Math.abs(ctxRms.biases[0][0] - (0.2 - updRB)) < 1e-6);

  // Adam update
  const ctxAdam = createCtx();
  ctxAdam.m_dw[0] = new Float32Array([0]);
  ctxAdam.v_dw[0] = new Float32Array([0]);
  ctxAdam.m_db[0] = new Float32Array([0]);
  ctxAdam.v_db[0] = new Float32Array([0]);
  oblixOptimizers.updateParameters(
    ctxAdam,
    [new Float32Array([0.1])],
    [new Float32Array([0.05])],
    [],
    [],
    {
      learningRate: 0.1,
      initialLearningRate: 0.1,
      optimizer: 'adam',
      batchSize: 1,
      l2Lambda: 0,
      gradientClipValue: 0,
      decayRate: 0.9,
    },
  );
  const m = 0.9 * 0 + 0.1 * 0.1;
  const v = 0.999 * 0 + 0.001 * 0.1 * 0.1;
  const mHat = m / (1 - Math.pow(0.9, 1));
  const vHat = v / (1 - Math.pow(0.999, 1));
  const baseLR = (0.1 * Math.sqrt(1 - Math.pow(0.999, 1))) / (1 - Math.pow(0.9, 1));
  const stepLR = baseLR * (0.1 / 0.1);
  const updA = (stepLR * mHat) / (Math.sqrt(vHat) + 1e-8);
  const mB = 0.9 * 0 + 0.1 * 0.05;
  const vB = 0.999 * 0 + 0.001 * 0.05 * 0.05;
  const mHatB = mB / (1 - Math.pow(0.9, 1));
  const vHatB = vB / (1 - Math.pow(0.999, 1));
  const updB = (stepLR * mHatB) / (Math.sqrt(vHatB) + 1e-8);
  assert.ok(Math.abs(ctxAdam.m_dw[0][0] - m) < 1e-6);
  assert.ok(Math.abs(ctxAdam.v_dw[0][0] - v) < 1e-6);
  assert.ok(Math.abs(ctxAdam.weights[0][0] - (0.5 - updA)) < 1e-6);
  assert.ok(Math.abs(ctxAdam.biases[0][0] - (0.2 - updB)) < 1e-6);
}

