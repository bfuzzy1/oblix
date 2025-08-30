import { RLAgent } from './agent.js';
import { mapToObject, objectToMap } from './utils/serialization.js';

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
    return this._selectAction(qVals, state, update);
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
    const data = super.toJSON();
    data.type = 'double';
    delete data.qTable;
    data.qTableA = mapToObject(this.qTableA);
    data.qTableB = mapToObject(this.qTableB);
    return data;
  }

  static fromJSON(data) {
    const base = RLAgent.fromJSON(data);
    const agent = new DoubleQAgent({
      epsilon: base.epsilon,
      gamma: base.gamma,
      learningRate: base.learningRate,
      epsilonDecay: base.epsilonDecay,
      minEpsilon: base.minEpsilon,
      policy: base.policy,
      temperature: base.temperature,
      ucbC: base.ucbC
    });
    agent.countTable = base.countTable;
    agent.qTableA = objectToMap(data.qTableA, Float32Array);
    agent.qTableB = objectToMap(data.qTableB, Float32Array);
    return agent;
  }
}
