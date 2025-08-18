export class RLAgent {
  constructor(options = {}) {
    this.epsilon = options.epsilon ?? 0.1;
    this.gamma = options.gamma ?? 0.95;
    this.learningRate = options.learningRate ?? 0.1;
    this.epsilonDecay = options.epsilonDecay ?? 0.99;
    this.minEpsilon = options.minEpsilon ?? 0.01;
    this.qTable = new Map();
  }

  _key(state) {
    return Array.from(state).join(',');
  }

  _ensure(state) {
    const key = this._key(state);
    if (!this.qTable.has(key)) {
      this.qTable.set(key, new Float32Array(4));
    }
    return this.qTable.get(key);
  }

  /** Choose an action using epsilon-greedy policy. */
  act(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 4);
    }
    const qVals = this._ensure(state);
    let best = 0;
    for (let i = 1; i < qVals.length; i++) {
      if (qVals[i] > qVals[best]) best = i;
    }
    return best;
  }

  /** Reduce exploration rate after learning. */
  decayEpsilon() {
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  /** Perform tabular Q-learning update. */
  learn(state, action, reward, nextState, done) {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    const maxNext = done ? 0 : Math.max(...nextQ);
    qVals[action] +=
      this.learningRate * (reward + this.gamma * maxNext - qVals[action]);
    this.decayEpsilon();
  }
}
