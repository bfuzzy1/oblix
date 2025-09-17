import { PlannerAgent } from './plannerAgent.js';

export class ValueIterationAgent extends PlannerAgent {
  constructor(options = {}) {
    super(options);
    this.convergenceTolerance = options.theta ?? options.convergenceTolerance ?? 1e-4;
    this.maxIterations = options.maxIterations ?? 1000;
  }

  planPolicy(environment = this.environment) {
    if (!environment) return;
    const states = environment.enumerateStates();
    if (states.length === 0) {
      this.policyMap.clear();
      this.valueFunction.clear();
      return;
    }
    const actions = environment.getAvailableActions();
    let values = new Map(this.valueFunction);
    for (const state of states) {
      const key = this._key(state);
      if (!values.has(key)) {
        values.set(key, 0);
      }
    }
    let iterations = 0;
    let delta = Infinity;
    while (delta > this.convergenceTolerance && iterations < this.maxIterations) {
      const updatedValues = new Map(values);
      delta = 0;
      for (const state of states) {
        const key = this._key(state);
        if (environment.isTerminalState(state)) {
          updatedValues.set(key, 0);
          continue;
        }
        let bestValue = -Infinity;
        for (const action of actions) {
          const transition = environment.getTransition(state, action);
          const nextKey = this._key(transition.state);
          const nextValue = values.get(nextKey) ?? 0;
          const candidate = transition.reward + this.gamma * (transition.done ? 0 : nextValue);
          if (candidate > bestValue) {
            bestValue = candidate;
          }
        }
        const newValue = bestValue === -Infinity ? 0 : bestValue;
        const previous = values.get(key) ?? 0;
        updatedValues.set(key, newValue);
        delta = Math.max(delta, Math.abs(newValue - previous));
      }
      values = updatedValues;
      iterations += 1;
    }
    this.valueFunction = values;
    this.policyMap = new Map();
    for (const state of states) {
      const key = this._key(state);
      if (environment.isTerminalState(state)) {
        this.policyMap.set(key, 0);
        continue;
      }
      let bestAction = 0;
      let bestValue = -Infinity;
      for (const action of actions) {
        const transition = environment.getTransition(state, action);
        const nextKey = this._key(transition.state);
        const nextValue = this.valueFunction.get(nextKey) ?? 0;
        const candidate = transition.reward + this.gamma * (transition.done ? 0 : nextValue);
        if (candidate > bestValue) {
          bestValue = candidate;
          bestAction = action;
        }
      }
      this.policyMap.set(key, bestAction);
    }
  }

  toJSON() {
    return {
      type: 'value-iteration',
      gamma: this.gamma,
      theta: this.convergenceTolerance,
      maxIterations: this.maxIterations
    };
  }

  static fromJSON(data = {}) {
    return new ValueIterationAgent({
      gamma: data.gamma,
      theta: data.theta,
      maxIterations: data.maxIterations
    });
  }
}
