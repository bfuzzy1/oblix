import * as tf from '@tensorflow/tfjs';
import { RLAgent } from './agent.js';

export class DQNAgent extends RLAgent {
  constructor(options = {}) {
    const opts = { learningRate: 0.001, ...options };
    super(opts);
    this.stateSize = options.stateSize ?? 2;
    this.actionSize = 4;
    this.bufferSize = options.bufferSize ?? 500;
    this.batchSize = options.batchSize ?? 32;
    this.buffer = [];
    this.optimizer = tf.train.adam(this.learningRate);
    this.model = this._buildModel();
  }

  _buildModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 24, activation: 'relu', inputShape: [this.stateSize] }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    model.add(tf.layers.dense({ units: this.actionSize }));
    return model;
  }

  act(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionSize);
    }
    return tf.tidy(() => {
      const input = tf.tensor2d([Array.from(state)]);
      const qVals = this.model.predict(input);
      const action = qVals.argMax(-1).dataSync()[0];
      return action;
    });
  }

  async learn(state, action, reward, nextState, done) {
    this.buffer.push({
      state: Array.from(state),
      action,
      reward,
      nextState: Array.from(nextState),
      done
    });
    if (this.buffer.length > this.bufferSize) {
      this.buffer.shift();
    }
    if (this.buffer.length < this.batchSize) {
      this.decayEpsilon();
      return;
    }
    const batch = [];
    for (let i = 0; i < this.batchSize; i++) {
      const idx = Math.floor(Math.random() * this.buffer.length);
      batch.push(this.buffer[idx]);
    }
    await tf.tidy(() => {
      const states = tf.tensor2d(batch.map(e => e.state));
      const nextStates = tf.tensor2d(batch.map(e => e.nextState));
      const rewards = tf.tensor1d(batch.map(e => e.reward));
      const dones = tf.tensor1d(batch.map(e => (e.done ? 0 : 1)));
      const actions = tf.tensor1d(batch.map(e => e.action), 'int32');
      const targetQs = this.model.predict(nextStates).max(1).mul(this.gamma).mul(dones).add(rewards);
      this.optimizer.minimize(() => {
        const qVals = this.model.predict(states);
        const masks = tf.oneHot(actions, this.actionSize);
        const pred = qVals.mul(masks).sum(1);
        const loss = tf.losses.meanSquaredError(targetQs, pred);
        return loss;
      });
    });
    await tf.nextFrame();
    this.decayEpsilon();
  }

  reset() {
    super.reset();
    this.buffer = [];
    this.model = this._buildModel();
  }
}
