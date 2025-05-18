import { oblixLayerOps } from '../src/layers.js';
import { performance } from 'perf_hooks';

export async function run() {
  const dims = 512;
  const input = new Float32Array(dims);
  for (let i = 0; i < dims; i++) input[i] = Math.random();
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixLayerOps.softmaxForward({}, input);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`softmaxForward: ${dims} dims -> avg ${avg.toFixed(2)} ms`);
}
