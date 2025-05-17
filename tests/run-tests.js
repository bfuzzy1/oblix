import assert from 'assert';
import { Oblix } from '../src/core/network.js';

const nn = new Oblix(false);
nn.layer({ type: 'dense', inputSize: 2, outputSize: 1, activation: 'relu', useBias: true });
const output = nn.predict([0.5, -0.5]);
assert(output instanceof Float32Array && output.length === 1, 'predict should return array of length 1');
console.log('All tests passed');
