import { RLAgent } from './agent.js';

export class QLambdaAgent extends RLAgent {
  constructor(options = {}) {
    super(options);
    this.lambda = options.lambda ?? 0.8;
    this.eligibility = new Map();
  }

  _ensureEligibility(state) {
    const key = this._key(state);
    if (!this.eligibility.has(key)) {
      this.eligibility.set(key, new Float32Array(4));
    }
    return this.eligibility.get(key);
  }

  learn(state, action, reward, nextState, done) {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    const elig = this._ensureEligibility(state);
    elig[action] += 1;
    const target = reward + (done ? 0 : this.gamma * Math.max(...nextQ));
    const delta = target - qVals[action];
    for (const [key, eVals] of this.eligibility.entries()) {
      const q = this.qTable.get(key);
      for (let i = 0; i < eVals.length; i++) {
        q[i] += this.learningRate * delta * eVals[i];
        eVals[i] *= this.lambda * this.gamma;
      }
    }
    if (done) this.eligibility.clear();
    this.decayEpsilon();
  }

  reset() {
    super.reset();
    this.eligibility.clear();
  }

  toJSON() {
    const data = super.toJSON();
    data.lambda = this.lambda;
    return data;
  }

  static fromJSON(data) {
    const agent = new QLambdaAgent({
      epsilon: data.epsilon,
      gamma: data.gamma,
      learningRate: data.learningRate,
      epsilonDecay: data.epsilonDecay,
      minEpsilon: data.minEpsilon,
      policy: data.policy,
      temperature: data.temperature,
      ucbC: data.ucbC,
      lambda: data.lambda
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
