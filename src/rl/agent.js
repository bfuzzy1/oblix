export class RLAgent {
  constructor(options = {}) {
    this.initialEpsilon = options.epsilon ?? 0.1;
    this.epsilon = this.initialEpsilon;
    this.gamma = options.gamma ?? 0.95;
    this.learningRate = options.learningRate ?? 0.1;
    this.epsilonDecay = options.epsilonDecay ?? 0.99;
    this.minEpsilon = options.minEpsilon ?? 0.01;
    this.policy = options.policy ?? 'epsilon-greedy';
    this.temperature = options.temperature ?? 1;
    this.ucbC = options.ucbC ?? 2;
    this.qTable = new Map();
    this.countTable = new Map();
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

  /** Choose an action using the configured policy. */
  act(state, update = true) {
    const qVals = this._ensure(state);
    switch (this.policy) {
      case 'greedy':
        return this._greedy(qVals);
      case 'softmax':
        return this._softmax(qVals);
      case 'ucb':
        return this._ucb(state, qVals, update);
      case 'epsilon-greedy':
      default:
        return this._epsilonGreedy(qVals);
    }
  }

  _random() {
    return Math.floor(Math.random() * 4);
  }

  _greedy(qVals) {
    let best = 0;
    for (let i = 1; i < qVals.length; i++) {
      if (qVals[i] > qVals[best]) best = i;
    }
    return best;
  }

  _epsilonGreedy(qVals) {
    if (Math.random() < this.epsilon) {
      return this._random();
    }
    return this._greedy(qVals);
  }

  _softmax(qVals) {
    const max = Math.max(...qVals);
    const exps = qVals.map(v => Math.exp((v - max) / this.temperature));
    const sum = exps.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < exps.length; i++) {
      r -= exps[i];
      if (r <= 0) return i;
    }
    return exps.length - 1;
  }

  _ensureCount(state) {
    const key = this._key(state);
    if (!this.countTable.has(key)) {
      this.countTable.set(key, new Uint32Array(4));
    }
    return this.countTable.get(key);
  }

  _ucb(state, qVals, update) {
    const counts = this._ensureCount(state);
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] === 0) {
        if (update) counts[i]++;
        return i;
      }
    }
    const total = counts.reduce((a, b) => a + b, 0);
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < qVals.length; i++) {
      const score = qVals[i] + this.ucbC * Math.sqrt(Math.log(total) / counts[i]);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    if (update) counts[best]++;
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
    qVals[action] += this.learningRate * (reward + this.gamma * maxNext - qVals[action]);
    this.decayEpsilon();
  }

  /** Reset agent to initial state. */
  reset() {
    this.epsilon = this.initialEpsilon;
    this.qTable.clear();
    this.countTable.clear();
  }

  /** Serialize agent state to a plain object. */
  toJSON() {
    const table = Object.fromEntries(
      Array.from(this.qTable.entries()).map(([k, v]) => [k, Array.from(v)])
    );
    const counts = Object.fromEntries(
      Array.from(this.countTable.entries()).map(([k, v]) => [k, Array.from(v)])
    );
    return {
      epsilon: this.epsilon,
      gamma: this.gamma,
      learningRate: this.learningRate,
      epsilonDecay: this.epsilonDecay,
      minEpsilon: this.minEpsilon,
      policy: this.policy,
      temperature: this.temperature,
      ucbC: this.ucbC,
      qTable: table,
      countTable: counts
    };
  }

  /** Recreate an agent from serialized data. */
  static fromJSON(data) {
    const agent = new RLAgent({
      epsilon: data.epsilon,
      gamma: data.gamma,
      learningRate: data.learningRate,
      epsilonDecay: data.epsilonDecay,
      minEpsilon: data.minEpsilon,
      policy: data.policy,
      temperature: data.temperature,
      ucbC: data.ucbC
    });
    for (const [k, v] of Object.entries(data.qTable || {})) {
      agent.qTable.set(k, new Float32Array(v));
    }
    for (const [k, v] of Object.entries(data.countTable || {})) {
      agent.countTable.set(k, new Uint32Array(v));
    }
    return agent;
  }
}
