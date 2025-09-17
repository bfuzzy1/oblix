import { PlannerAgent } from './plannerAgent.js';

export class PolicyIterationAgent extends PlannerAgent {
  constructor(options = {}) {
    super(options);
    this.evaluationTolerance = options.theta ?? options.evaluationTolerance ?? 1e-4;
    this.maxEvaluationIterations = options.maxEvaluationIterations ?? 100;
    this.maxPolicyIterations = options.maxPolicyIterations ?? 100;
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
    let policy = new Map(this.policyMap);
    for (const state of states) {
      const key = this._key(state);
      if (environment.isTerminalState(state)) {
        policy.set(key, 0);
        continue;
      }
      if (!policy.has(key)) {
        policy.set(key, actions[0] ?? 0);
      }
    }
    let values = new Map(this.valueFunction);
    for (const state of states) {
      const key = this._key(state);
      if (!values.has(key)) {
        values.set(key, 0);
      }
    }
    let isStable = false;
    let iterations = 0;
    while (!isStable && iterations < this.maxPolicyIterations) {
      values = this._evaluatePolicy(environment, states, actions, policy, values);
      isStable = this._improvePolicy(environment, states, actions, policy, values);
      iterations += 1;
    }
    this.policyMap = new Map(policy);
    this.valueFunction = new Map(values);
  }

  _evaluatePolicy(environment, states, actions, policy, initialValues) {
    let values = new Map(initialValues);
    let iterations = 0;
    let delta = Infinity;
    while (delta > this.evaluationTolerance && iterations < this.maxEvaluationIterations) {
      const updated = new Map(values);
      delta = 0;
      for (const state of states) {
        const key = this._key(state);
        if (environment.isTerminalState(state)) {
          updated.set(key, 0);
          continue;
        }
        const action = policy.get(key) ?? actions[0] ?? 0;
        const transition = environment.getTransition(state, action);
        const nextKey = this._key(transition.state);
        const nextValue = values.get(nextKey) ?? 0;
        const newValue = transition.reward + this.gamma * (transition.done ? 0 : nextValue);
        const previous = values.get(key) ?? 0;
        updated.set(key, newValue);
        delta = Math.max(delta, Math.abs(newValue - previous));
      }
      values = updated;
      iterations += 1;
    }
    return values;
  }

  _improvePolicy(environment, states, actions, policy, values) {
    let stable = true;
    for (const state of states) {
      const key = this._key(state);
      if (environment.isTerminalState(state)) {
        policy.set(key, 0);
        continue;
      }
      const currentAction = policy.get(key) ?? actions[0] ?? 0;
      let bestAction = currentAction;
      let bestValue = -Infinity;
      for (const action of actions) {
        const transition = environment.getTransition(state, action);
        const nextKey = this._key(transition.state);
        const nextValue = values.get(nextKey) ?? 0;
        const candidate = transition.reward + this.gamma * (transition.done ? 0 : nextValue);
        if (candidate > bestValue) {
          bestValue = candidate;
          bestAction = action;
        }
      }
      if (bestAction !== currentAction) {
        stable = false;
      }
      policy.set(key, bestAction);
    }
    return stable;
  }

  toJSON() {
    return {
      type: 'policy-iteration',
      gamma: this.gamma,
      theta: this.evaluationTolerance,
      maxEvaluationIterations: this.maxEvaluationIterations,
      maxPolicyIterations: this.maxPolicyIterations
    };
  }

  static fromJSON(data = {}) {
    return new PolicyIterationAgent({
      gamma: data.gamma,
      theta: data.theta,
      maxEvaluationIterations: data.maxEvaluationIterations,
      maxPolicyIterations: data.maxPolicyIterations
    });
  }
}
