import { oblixUtils } from '../src/utils.js';
import { performance } from 'perf_hooks';

export async function run() {
  const samples = 10000;
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixUtils.generateGaussianBlobs(samples, 0.1, 3);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(
    `generateGaussianBlobs: ${samples} samples -> avg ${avg.toFixed(2)} ms`,
  );
}
