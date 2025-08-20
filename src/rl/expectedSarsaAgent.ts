import { RLAgent } from './agent.js';

export class ExpectedSarsaAgent extends RLAgent {
  learn(
    state: Float32Array,
    action: number,
    reward: number,
    nextState: Float32Array,
    done: boolean
  ): void | Promise<void> {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    let expected = 0;
    if (!done) {
      let best = 0;
      for (let i = 1; i < nextQ.length; i++) {
        if (nextQ[i] > nextQ[best]) best = i;
      }
      const numActions = nextQ.length;
      const epsPart = this.epsilon / numActions;
      for (let i = 0; i < numActions; i++) {
        let prob = epsPart;
        if (i === best) prob += 1 - this.epsilon;
        expected += prob * nextQ[i];
      }
    }
    const target = reward + (done ? 0 : this.gamma * expected);
    qVals[action] += this.learningRate * (target - qVals[action]);
    this.decayEpsilon();
  }
}
