import { RLAgent } from './agent.js';

/**
 * RL agent with optimistic initial values to encourage exploration.
 */
export class OptimisticAgent extends RLAgent {
  constructor(options = {}) {
    super(options);
    this.initialValue = options.initialValue ?? 1;
  }

  _ensure(state) {
    const key = this._key(state);
    if (!this.qTable.has(key)) {
      const arr = new Float32Array(4);
      arr.fill(this.initialValue);
      this.qTable.set(key, arr);
    }
    return this.qTable.get(key);
  }

  toJSON() {
    const data = super.toJSON();
    data.initialValue = this.initialValue;
    return data;
  }

  static fromJSON(data) {
    const base = RLAgent.fromJSON(data);
    const agent = new OptimisticAgent({
      epsilon: base.epsilon,
      gamma: base.gamma,
      learningRate: base.learningRate,
      epsilonDecay: base.epsilonDecay,
      minEpsilon: base.minEpsilon,
      policy: base.policy,
      temperature: base.temperature,
      ucbC: base.ucbC,
      initialValue: data.initialValue
    });
    agent.qTable = base.qTable;
    agent.countTable = base.countTable;
    return agent;
  }
}
