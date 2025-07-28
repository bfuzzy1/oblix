import assert from 'assert';
import { oblixLayerOps } from '../src/layers.js';

/**
 *
 */
export async function run() {
  const ctxF = { debug: false, epsilon: 1e-5, forwardCache: { activations: [0], layerNormIntermediates: [] } };
  const input = new Float32Array([1, 2, 3]);
  const gamma = new Float32Array([1, 1, 1]);
  const beta = new Float32Array([0, 0, 0]);
  const cache = oblixLayerOps.layerNormForward(ctxF, input, gamma, beta);
  const expectedNorm = [-1.2247357, 0, 1.2247357];
  for (let i = 0; i < expectedNorm.length; i++) {
    assert.ok(Math.abs(cache.normalizedInput[i] - expectedNorm[i]) < 1e-4);
    assert.ok(Math.abs(cache.output[i] - expectedNorm[i]) < 1e-4);
  }
  const dOutput = new Float32Array([0.1, 0.2, 0.3]);
  const back = oblixLayerOps.layerNormBackward({ debug: false, epsilon: 1e-5 }, dOutput, cache);
  assert.strictEqual(back.dInput.length, input.length);
  assert.strictEqual(back.dGamma.length, input.length);
  assert.strictEqual(back.dBeta.length, input.length);
  for (let i = 0; i < input.length; i++) {
    assert.ok(Math.abs(back.dGamma[i] - dOutput[i] * cache.normalizedInput[i]) < 1e-6);
    assert.ok(Math.abs(back.dBeta[i] - dOutput[i]) < 1e-6);
  }
  const eps = 1e-3;
  const numericGrad = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const plus = input.slice();
    plus[i] += eps;
    const minus = input.slice();
    minus[i] -= eps;
    const outPlus = oblixLayerOps.layerNormForward({ debug: false, epsilon: 1e-5, forwardCache: { activations: [0], layerNormIntermediates: [] } }, plus, gamma, beta).output;
    const outMinus = oblixLayerOps.layerNormForward({ debug: false, epsilon: 1e-5, forwardCache: { activations: [0], layerNormIntermediates: [] } }, minus, gamma, beta).output;
    let grad = 0;
    for (let j = 0; j < dOutput.length; j++) {
      grad += (outPlus[j] - outMinus[j]) * dOutput[j];
    }
    numericGrad[i] = grad / (2 * eps);
  }
  for (let i = 0; i < input.length; i++) {
    assert.ok(Math.abs(back.dInput[i] - numericGrad[i]) < 1e-3);
  }
}
