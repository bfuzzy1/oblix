import assert from 'assert';
import { oblixLayerOps } from '../src/layers.js';

/**
 *
 */
export async function run() {
  const ctxF = { debug: false, forwardCache: { activations: [0], softmaxOutputs: [] } };
  const input = new Float32Array([1, 2, 3]);
  const out = oblixLayerOps.softmaxForward(ctxF, input);
  const expected = [0.09003057, 0.24472848, 0.66524094];
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(out[i] - expected[i]) < 1e-4);
  }
  const cached = ctxF.forwardCache.softmaxOutputs[0];
  assert.ok(cached && cached.length === input.length);
  const dOut = new Float32Array([0.1, 0.2, 0.3]);
  const dIn = oblixLayerOps.softmaxBackward(ctxF, dOut, 0);
  assert.deepStrictEqual(Array.from(dIn), Array.from(dOut));
}
