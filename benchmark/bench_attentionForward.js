import { oblixLayerOps } from '../src/layers.js';
import { performance } from 'perf_hooks';

export async function run() {
  const numHeads = 8;
  const dims = 128; // divisible by numHeads
  const input = new Float32Array(dims);
  for (let i = 0; i < dims; i++) input[i] = Math.random();

  const ctx = { debug: false, forwardCache: { activations: [0], attentionIntermediates: [] } };
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixLayerOps.attentionForward(ctx, input, numHeads);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`attentionForward: ${dims}-dim, ${numHeads} heads -> avg ${avg.toFixed(2)} ms`);
}
