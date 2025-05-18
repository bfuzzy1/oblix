import { oblixUtils } from '../src/utils.js';
import { performance } from 'perf_hooks';

function prepData(samples, classes) {
  const preds = [];
  const targets = [];
  for (let i = 0; i < samples; i++) {
    const pred = [];
    for (let j = 0; j < classes; j++) pred.push(Math.random());
    preds.push(pred);
    const t = new Array(classes).fill(0);
    t[Math.floor(Math.random() * classes)] = 1;
    targets.push(t);
  }
  return { preds, targets };
}

export async function run() {
  const samples = 10000;
  const classes = 3;
  const { preds, targets } = prepData(samples, classes);
  const runs = 5;
  const times = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    oblixUtils.calculateAccuracy(preds, targets);
    const end = performance.now();
    times.push(end - start);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`calculateAccuracy: ${samples} samples -> avg ${avg.toFixed(2)} ms`);
}
