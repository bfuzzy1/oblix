import { RLAgent } from './agent.js';

export class DynaQAgent extends RLAgent {
  constructor(options = {}) {
    super(options);
    this.planningSteps = options.planningSteps ?? 5;
    this.model = new Map();
    this.stateActions = [];
  }

  _saKey(state, action) {
    return `${this._key(state)}|${action}`;
  }

  /**
   * Update the model with an observed transition.
   * @param {ArrayLike<number>} state - state before the action
   * @param {number} action - action taken
   * @param {number} reward - reward received
   * @param {ArrayLike<number>} nextState - state after the action
   * @param {boolean} done - whether the transition ended the episode
   * @returns {{state: Float32Array, nextState: Float32Array}} processed states
   */
  _updateModel(state, action, reward, nextState, done) {
    const s = new Float32Array(state);
    const ns = new Float32Array(nextState);
    const key = this._saKey(s, action);
    if (!this.model.has(key)) {
      this.stateActions.push({ state: s, action });
    }
    this.model.set(key, { nextState: ns, reward, done });
    return { state: s, nextState: ns };
  }

  /**
   * Update Q-values based on real experience.
   * @param {Float32Array} state - current state
   * @param {number} action - action taken
   * @param {number} reward - reward received
   * @param {Float32Array} nextState - state after the action
   * @param {boolean} done - whether the transition ended the episode
   */
  _updateQValues(state, action, reward, nextState, done) {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    const maxNext = done ? 0 : Math.max(...nextQ);
    qVals[action] += this.learningRate * (reward + this.gamma * maxNext - qVals[action]);
  }

  /**
   * Execute planning steps using the learned model.
   */
  _runPlanningSteps() {
    const total = this.stateActions.length;
    for (let i = 0; i < this.planningSteps && total > 0; i++) {
      const idx = Math.floor(Math.random() * total);
      const { state: ps, action: pa } = this.stateActions[idx];
      const modelKey = this._saKey(ps, pa);
      const modelEntry = this.model.get(modelKey);
      const pns = modelEntry.nextState;
      const pr = modelEntry.reward;
      const pd = modelEntry.done;
      const pq = this._ensure(ps);
      const pnextQ = this._ensure(pns);
      const pmaxNext = pd ? 0 : Math.max(...pnextQ);
      pq[pa] += this.learningRate * (pr + this.gamma * pmaxNext - pq[pa]);
    }
  }

  learn(state, action, reward, nextState, done) {
    const { state: s, nextState: ns } = this._updateModel(
      state,
      action,
      reward,
      nextState,
      done
    );
    this._updateQValues(s, action, reward, ns, done);
    this._runPlanningSteps();
    this.decayEpsilon();
  }
}
