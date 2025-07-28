import assert from 'assert';
import { oblixLayerOps } from '../src/layers.js';

/**
 *
 */
export async function run() {
  const ctxF = { debug: false, forwardCache: { activations: [0], attentionIntermediates: [] } };
  const input = new Float32Array([1, 2, 3, 4]);
  const output = oblixLayerOps.attentionForward(ctxF, input, 2);
  const expected = [1.6697615, 1.8044296, 3.8929582, 3.9441929];
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(output[i] - expected[i]) < 1e-4);
  }
  const cache = ctxF.forwardCache.attentionIntermediates[0];
  assert.ok(cache && cache.input instanceof Float32Array);
  const dOutput = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const { dInput } = oblixLayerOps.attentionBackward({ debug: false }, dOutput, cache);
  assert.strictEqual(dInput.length, input.length);
  const eps = 1e-3;
  const numericGrad = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const plus = input.slice();
    plus[i] += eps;
    const minus = input.slice();
    minus[i] -= eps;
    const outPlus = oblixLayerOps.attentionForward({ debug: false, forwardCache: { activations: [0], attentionIntermediates: [] } }, plus, 2);
    const outMinus = oblixLayerOps.attentionForward({ debug: false, forwardCache: { activations: [0], attentionIntermediates: [] } }, minus, 2);
    let grad = 0;
    for (let j = 0; j < dOutput.length; j++) {
      grad += (outPlus[j] - outMinus[j]) * dOutput[j];
    }
    numericGrad[i] = grad / (2 * eps);
  }
  for (let i = 0; i < input.length; i++) {
    assert.ok(Math.abs(dInput[i] - numericGrad[i]) < 1e-3);
  }
}
