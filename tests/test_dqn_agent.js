import * as tf from '@tensorflow/tfjs';
import { DQNAgent } from '../src/rl/dqnAgent.js';

export async function run(assert) {
  const agent = new DQNAgent({ epsilon: 0, bufferSize: 10, batchSize: 1 });
  const state = new Float32Array([0, 0]);
  const nextState = new Float32Array([0, 0]);

  const before = tf.tidy(() => {
    const q = agent.model.predict(tf.tensor2d([Array.from(state)]));
    return q.dataSync()[0];
  });

  for (let i = 0; i < 20; i++) {
    await agent.learn(state, 0, 1, nextState, true);
  }

  const after = tf.tidy(() => {
    const q = agent.model.predict(tf.tensor2d([Array.from(state)]));
    return q.dataSync()[0];
  });

  assert.ok(after > before);

  const agent2 = new DQNAgent({ bufferSize: 5, batchSize: 1 });
  for (let i = 0; i < 12; i++) {
    await agent2.learn(state, 0, 0, nextState, false);
  }
  assert.strictEqual(agent2.buffer.length, 5);
}
