import { oblixUtils } from '../src/utils.js';
import { performance } from 'perf_hooks';

export async function run() {
  const samples = 1000000;
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    for (let j = 0; j < samples; j++) {
      oblixUtils.gaussianRandom();
    }
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`gaussianRandom: ${samples} calls -> avg ${avg.toFixed(2)} ms`);
}
