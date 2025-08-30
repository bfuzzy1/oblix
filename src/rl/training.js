import { RLAgent } from './agent.js';
import { GridWorldEnvironment } from './environment.js';
import { ExperienceReplay } from './experienceReplay.js';

export class RLTrainer {
  constructor(agent, env, options = {}) {
    this.agent = agent;
    this.env = env;
    this.maxSteps = options.maxSteps ?? 50;
    this.intervalMs = options.intervalMs ?? 100;
    this.liveChart = options.liveChart || null;
    const userOnStep = options.onStep || null;
    this.onStep = (state, reward, done, metrics) => {
      if (userOnStep) userOnStep(state, reward, done, metrics);
      if (this.liveChart) {
        this.liveChart.push(metrics.cumulativeReward, metrics.epsilon);
      }
    };
    this.replaySamples = options.replaySamples ?? 0;
    this.replayStrategy = options.replayStrategy || 'uniform';
    this.replayBuffer = options.replayBuffer || (this.replaySamples > 0 ? new ExperienceReplay(
      options.bufferCapacity ?? 1000,
      options.bufferAlpha ?? 0.6,
      options.bufferBeta ?? 0.4
    ) : null);
    this.isRunning = false;
    this.interval = null;
    this.state = null;
    this.metrics = {
      episode: 1,
      steps: 0,
      cumulativeReward: 0,
      epsilon: this.agent.epsilon
    };
    this.episodeRewards = [];
  }

  async _applyTransition() {
    const action = await this.agent.act(this.state);
    const { state: nextState, reward, done } = this.env.step(action);
    const transition = { state: this.state, action, reward, nextState, done };
    await this.agent.learn(transition.state, transition.action, transition.reward, transition.nextState, transition.done);
    this.state = nextState;
    return transition;
  }

  async _processReplay(transition) {
    if (!this.replayBuffer) return;
    this.replayBuffer.add(transition, 1);
    const samples = this.replayBuffer.sample(this.replaySamples, this.replayStrategy);
    for (const t of samples) {
      await this.agent.learn(t.state, t.action, t.reward, t.nextState, t.done);
    }
  }

  _updateMetrics({ reward, done }) {
    this.metrics.steps += 1;
    this.metrics.cumulativeReward += reward;
    this.metrics.epsilon = this.agent.epsilon;
    if (this.onStep) this.onStep(this.state, reward, done, { ...this.metrics });
  }

  _handleEpisodeEnd(done) {
    if (!done) return;
    this.episodeRewards.push(this.metrics.cumulativeReward);
    this.metrics.episode += 1;
    this.metrics.steps = 0;
    this.metrics.cumulativeReward = 0;
    this.metrics.epsilon = this.agent.epsilon;
    this.state = this.env.reset();
    if (this.onStep) {
      this.onStep(this.state, 0, false, { ...this.metrics });
    }
  }

  async step() {
    if (!this.state) return;
    const transition = await this._applyTransition();
    await this._processReplay(transition);
    this._updateMetrics(transition);
    this._handleEpisodeEnd(transition.done);
  }

  start() {
    if (this.isRunning) return;
    this.state = this.env.reset();
    this.isRunning = true;
    this.interval = setInterval(async () => {
      await this.step();
    }, this.intervalMs);
  }

  setIntervalMs(ms) {
    this.intervalMs = ms;
    if (this.isRunning) {
      if (this.interval) clearInterval(this.interval);
      this.interval = setInterval(async () => {
        await this.step();
      }, this.intervalMs);
    }
  }

  pause() {
    if (!this.isRunning) return;
    if (this.interval) clearInterval(this.interval);
    this.isRunning = false;
  }

  reset() {
    this.pause();
    if (typeof this.agent.reset === 'function') {
      this.agent.reset();
    }
    this.state = this.env.reset();
    this.metrics = {
      episode: 1,
      steps: 0,
      cumulativeReward: 0,
      epsilon: this.agent.epsilon
    };
    this.episodeRewards = [];
    if (this.onStep) {
      this.onStep(this.state, 0, false, { ...this.metrics });
    }
  }

  static async trainEpisodes(agent, env, episodes = 10, maxSteps = 50) {
    for (let ep = 0; ep < episodes; ep++) {
      let state = env.reset();
      for (let st = 0; st < maxSteps; st++) {
        const action = await agent.act(state);
        const { state: nextState, reward, done } = env.step(action);
        await agent.learn(state, action, reward, nextState, done);
        state = nextState;
        if (done) break;
      }
    }
  }
}
