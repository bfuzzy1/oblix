import { oblixLayerOps } from '../src/layers.js';
import { performance } from 'perf_hooks';

export async function run() {
  const size = 10000; // Reduced from 1000000 to avoid crypto API limits
  const input = new Float32Array(size).fill(1);
  const ctx = { isTraining: true, debug: false, masks: [], forwardCache: { activations: [0] } };
  const rate = 0.5;
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixLayerOps.dropoutForward(ctx, input, rate);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`dropoutForward: ${size} elems -> avg ${avg.toFixed(2)} ms`);
}
