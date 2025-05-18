import { oblixLayerOps } from '../src/layers.js';
import { performance } from 'perf_hooks';

function buildInput(size) {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) arr[i] = Math.random();
  return arr;
}

export async function run() {
  const dims = 256;
  const iterations = 1000;
  const input = buildInput(dims);
  const gamma = new Float32Array(dims).fill(1);
  const beta = new Float32Array(dims).fill(0);
  const ctx = { debug: false, epsilon: 1e-5, forwardCache: { activations: [0], layerNormIntermediates: [] } };
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    for (let j = 0; j < iterations; j++) {
      oblixLayerOps.layerNormForward(ctx, input, gamma, beta);
    }
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`layerNormForward: ${iterations} iterations (${dims} dims) -> avg ${avg.toFixed(2)} ms`);
}
