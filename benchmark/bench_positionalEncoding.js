import { oblixUtils } from '../src/utils.js';
import { performance } from 'perf_hooks';

export async function run() {
  const dims = 256;
  const input = new Float32Array(dims).fill(0.5);
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixUtils.positionalEncoding(input);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`positionalEncoding: ${dims} dims -> avg ${avg.toFixed(2)} ms`);
}
