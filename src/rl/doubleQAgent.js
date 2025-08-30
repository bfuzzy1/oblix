import { RLAgent } from './agent.js';

export class DoubleQAgent extends RLAgent {
  constructor(options = {}) {
    super(options);
    this.qTableA = new Map();
    this.qTableB = new Map();
  }

  _ensure(table, state) {
    const key = this._key(state);
    if (!table.has(key)) {
      table.set(key, new Float32Array(4));
    }
    return table.get(key);
  }

  _ensureBoth(state) {
    const qa = this._ensure(this.qTableA, state);
    const qb = this._ensure(this.qTableB, state);
    return [qa, qb];
  }

  act(state, update = true) {
    const [qa, qb] = this._ensureBoth(state);
    const qVals = qa.map((v, i) => (v + qb[i]) / 2);
    switch (this.policy) {
      case 'greedy':
        return this._greedy(qVals);
      case 'softmax':
        return this._softmax(qVals);
      case 'thompson':
        return this._thompson(state, qVals, update);
      case 'ucb':
        return this._ucb(state, qVals, update);
      case 'epsilon-greedy':
      default:
        return this._epsilonGreedy(qVals);
    }
  }

  learn(state, action, reward, nextState, done) {
    const updateA = Math.random() < 0.5;
    const [qa, qb] = this._ensureBoth(state);
    const [nextQa, nextQb] = this._ensureBoth(nextState);
    if (updateA) {
      let best = 0;
      for (let i = 1; i < nextQa.length; i++) {
        if (nextQa[i] > nextQa[best]) best = i;
      }
      const target = reward + (done ? 0 : this.gamma * nextQb[best]);
      qa[action] += this.learningRate * (target - qa[action]);
    } else {
      let best = 0;
      for (let i = 1; i < nextQb.length; i++) {
        if (nextQb[i] > nextQb[best]) best = i;
      }
      const target = reward + (done ? 0 : this.gamma * nextQa[best]);
      qb[action] += this.learningRate * (target - qb[action]);
    }
    this.decayEpsilon();
  }

  reset() {
    super.reset();
    this.qTableA.clear();
    this.qTableB.clear();
  }

  toJSON() {
    const tableA = Object.fromEntries(
      Array.from(this.qTableA.entries()).map(([k, v]) => [k, Array.from(v)])
    );
    const tableB = Object.fromEntries(
      Array.from(this.qTableB.entries()).map(([k, v]) => [k, Array.from(v)])
    );
    const counts = Object.fromEntries(
      Array.from(this.countTable.entries()).map(([k, v]) => [k, Array.from(v)])
    );
    return {
      type: 'double',
      epsilon: this.epsilon,
      gamma: this.gamma,
      learningRate: this.learningRate,
      epsilonDecay: this.epsilonDecay,
      minEpsilon: this.minEpsilon,
      policy: this.policy,
      temperature: this.temperature,
      ucbC: this.ucbC,
      qTableA: tableA,
      qTableB: tableB,
      countTable: counts
    };
  }

  static fromJSON(data) {
    const agent = new DoubleQAgent({
      epsilon: data.epsilon,
      gamma: data.gamma,
      learningRate: data.learningRate,
      epsilonDecay: data.epsilonDecay,
      minEpsilon: data.minEpsilon,
      policy: data.policy,
      temperature: data.temperature,
      ucbC: data.ucbC
    });
    for (const [k, v] of Object.entries(data.qTableA || {})) {
      agent.qTableA.set(k, new Float32Array(v));
    }
    for (const [k, v] of Object.entries(data.qTableB || {})) {
      agent.qTableB.set(k, new Float32Array(v));
    }
    for (const [k, v] of Object.entries(data.countTable || {})) {
      agent.countTable.set(k, new Uint32Array(v));
    }
    return agent;
  }
}
