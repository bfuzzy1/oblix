import { Oblix } from "../network.js";

export class RLAgent {
  constructor(nn, options = {}) {
    this.nn = nn;
    this.epsilon = options.epsilon ?? 0.1;
    this.gamma = options.gamma ?? 0.95;
    this.learningRate = options.learningRate ?? 0.01;
  }

  /** Choose an action using epsilon-greedy policy. */
  act(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 4);
    }
    const qVals = this.nn.predict(state);
    if (!qVals) return Math.floor(Math.random() * 4);
    let best = 0;
    for (let i = 1; i < qVals.length; i++) {
      if (qVals[i] > qVals[best]) best = i;
    }
    return best;
  }

  /** Perform Q-learning update using network. */
  async learn(state, action, reward, nextState, done) {
    const qVals = this.nn.predict(state);
    const nextQ = this.nn.predict(nextState);
    if (!qVals || !nextQ) return;
    const target = new Float32Array(qVals);
    const maxNext = Math.max(...nextQ);
    target[action] = reward + (done ? 0 : this.gamma * maxNext);
    const dataset = [{ input: state, output: target }];
    await this.nn.train(dataset, {
      epochs: 1,
      batchSize: 1,
      learningRate: this.learningRate,
      optimizer: "sgd",
      lossFunction: "mse",
    });
  }
}
