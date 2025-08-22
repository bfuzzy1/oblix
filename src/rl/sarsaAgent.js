import { RLAgent } from './agent.js';

export class SarsaAgent extends RLAgent {
  learn(state, action, reward, nextState, done) {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    let nextAction = 0;
    if (!done) {
      nextAction = this.act(nextState);
    }
    const target = reward + (done ? 0 : this.gamma * nextQ[nextAction]);
    qVals[action] += this.learningRate * (target - qVals[action]);
    this.decayEpsilon();
  }
}
