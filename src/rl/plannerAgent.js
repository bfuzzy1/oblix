import { RLAgent } from './agent.js';

export class PlannerAgent extends RLAgent {
  constructor(options = {}) {
    super({
      ...options,
      epsilon: 0,
      epsilonDecay: 1,
      minEpsilon: 0
    });
    this.initialEpsilon = 0;
    this.epsilon = 0;
    this.learningRate = undefined;
    this.policyMap = new Map();
    this.valueFunction = new Map();
    this.environment = null;
  }

  setEnvironment(environment) {
    this.environment = environment || null;
  }

  reset(environment) {
    if (environment) {
      this.setEnvironment(environment);
    }
    super.reset();
    this.epsilon = 0;
    this.policyMap.clear();
    this.valueFunction.clear();
    if (this.environment) {
      this.planPolicy(this.environment);
    }
  }

  planPolicy() {
    throw new Error('planPolicy() must be implemented by planner agents');
  }

  act(state) {
    const key = this._key(state);
    if (!this.policyMap.has(key) && this.environment) {
      this.planPolicy(this.environment);
    }
    if (!this.policyMap.has(key)) {
      return 0;
    }
    return this.policyMap.get(key);
  }

  learn() {
    return 0;
  }
}
