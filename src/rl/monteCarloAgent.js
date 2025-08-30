import { RLAgent } from './agent.js';

export class MonteCarloAgent extends RLAgent {
  constructor(options = {}) {
    super(options);
    this.exploringStarts = options.exploringStarts ?? false;
    this.episode = [];
  }

  act(state, update = true) {
    if (this.exploringStarts && this.episode.length === 0) {
      return this._random();
    }
    return super.act(state, update);
  }

  learn(state, action, reward, nextState, done) {
    this.episode.push({
      state: new Float32Array(state),
      key: this._key(state),
      action,
      reward
    });
    if (done) {
      let G = 0;
      for (let i = this.episode.length - 1; i >= 0; i--) {
        const step = this.episode[i];
        G = step.reward + this.gamma * G;
        let firstVisit = true;
        for (let j = 0; j < i; j++) {
          const prev = this.episode[j];
          if (prev.key === step.key && prev.action === step.action) {
            firstVisit = false;
            break;
          }
        }
        if (firstVisit) {
          const qVals = this._ensure(step.state);
          const counts = this._ensureCount(step.state);
          counts[step.action] += 1;
          qVals[step.action] += (G - qVals[step.action]) / counts[step.action];
        }
      }
      this.episode = [];
      this.decayEpsilon();
    }
  }

  reset() {
    super.reset();
    this.episode = [];
  }
}

