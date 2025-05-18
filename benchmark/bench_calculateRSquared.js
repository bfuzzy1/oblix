import { oblixUtils } from '../src/utils.js';
import { performance } from 'perf_hooks';

function prepData(samples) {
  const targets = [];
  const preds = [];
  for (let i = 0; i < samples; i++) {
    const t = Math.random();
    targets.push(t);
    preds.push(t + (Math.random() - 0.5) * 0.1);
  }
  return { preds, targets };
}

export async function run() {
  const samples = 10000;
  const { preds, targets } = prepData(samples);
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixUtils.calculateRSquared(preds, targets);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`calculateRSquared: ${samples} samples -> avg ${avg.toFixed(2)} ms`);
}
