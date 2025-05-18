import { oblixActivations } from '../src/activations.js';
import { performance } from 'perf_hooks';

export async function run() {
  const activations = [
    'relu',
    'tanh',
    'sigmoid',
    'leakyrelu',
    'gelu',
    'selu',
    'swish',
    'mish',
  ];
  const samples = 1000000;
  const data = new Array(samples);
  for (let i = 0; i < samples; i++) {
    data[i] = Math.random() * 4 - 2;
  }
  const runs = 5;
  for (const act of activations) {
    const times = [];
    for (let i = 0; i < runs; i++) {
      const start = performance.now();
      for (let j = 0; j < samples; j++) {
        oblixActivations.apply(data[j], act);
      }
      const end = performance.now();
      times.push(end - start);
    }
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    console.log(
      `activation.apply ${act}: ${samples} values -> avg ${avg.toFixed(2)} ms`,
    );
  }
}
