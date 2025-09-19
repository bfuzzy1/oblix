import { mapToObject, objectToMap } from './utils/serialization.js';
import { POLICIES, selectAction } from './policies.js';

export class RLAgent {
  constructor(options = {}) {
    this.initialEpsilon = options.epsilon ?? 0.1;
    this.epsilon = this.initialEpsilon;
    this.gamma = options.gamma ?? 0.95;
    this.learningRate = options.learningRate ?? 0.1;
    this.epsilonDecay = options.epsilonDecay ?? 0.99;
    this.minEpsilon = options.minEpsilon ?? 0.01;
    this.policy = options.policy ?? POLICIES.EPSILON_GREEDY;
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

  /**
   * Select an action using the configured policy.
   * @protected
   */
  _selectAction(qVals, state, update = true) {
    return selectAction(this.policy, this, state, qVals, update);
  }

  /** Choose an action using the configured policy. */
  act(state, update = true) {
    const qVals = this._ensure(state);
    return this._selectAction(qVals, state, update);
  }

  _random() {
    return Math.floor(Math.random() * 4);
  }

  bestAction(qVals) {
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
    return this.bestAction(qVals);
  }

  _softmax(qVals) {
    if (!(this.temperature > 0)) {
      return this.bestAction(qVals);
    }
    const max = Math.max(...qVals);
    const exps = qVals.map(v => Math.exp((v - max) / this.temperature));
    const sum = exps.reduce((a, b) => a + b, 0);
    if (!isFinite(sum) || sum === 0) {
      return this.bestAction(qVals);
    }
    let cumulative = 0;
    const r = Math.random();
    for (let i = 0; i < exps.length; i++) {
      cumulative += exps[i] / sum;
      if (r <= cumulative) return i;
    }
    return exps.length - 1;
  }

  _gaussian() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  _thompson(state, qVals, update) {
    const counts = this._ensureCount(state);
    let best = 0;
    let bestSample = -Infinity;
    for (let i = 0; i < qVals.length; i++) {
      const variance = 1 / (counts[i] + 1);
      const sample = qVals[i] + this._gaussian() * Math.sqrt(variance);
      if (sample > bestSample) {
        bestSample = sample;
        best = i;
      }
    }
    if (update) counts[best]++;
    return best;
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

  /**
   * Compute the temporal-difference error for a transition.
   * Agents can override this to customize replay priority updates.
   */
  computeTdError(state, action, reward, nextState, done) {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    const maxNext = done ? 0 : Math.max(...nextQ);
    const target = reward + this.gamma * maxNext;
    return target - qVals[action];
  }

  /** Reset agent to initial state. */
  reset() {
    this.epsilon = this.initialEpsilon;
    this.qTable.clear();
    this.countTable.clear();
  }

  /** Serialize agent state to a plain object. */
  toJSON() {
    return {
      type: 'rl',
      epsilon: this.epsilon,
      gamma: this.gamma,
      learningRate: this.learningRate,
      epsilonDecay: this.epsilonDecay,
      minEpsilon: this.minEpsilon,
      policy: this.policy,
      temperature: this.temperature,
      ucbC: this.ucbC,
      qTable: mapToObject(this.qTable),
      countTable: mapToObject(this.countTable)
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
    agent.qTable = objectToMap(data.qTable, Float32Array);
    agent.countTable = objectToMap(data.countTable, Uint32Array);
    return agent;
  }
}
