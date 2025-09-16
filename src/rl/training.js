import { RLAgent } from './agent.js';
import { GridWorldEnvironment } from './environment.js';
import { ExperienceReplay } from './experienceReplay.js';
import { MetricsTracker } from './metricsTracker.js';

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
    this.isStepping = false; // prevents overlapping step calls
    this.timeout = null;
    this.state = null;
    this.metricsTracker = new MetricsTracker(this.agent);
    this.metrics = this.metricsTracker.data;
    this.episodeRewards = this.metricsTracker.episodeRewards;
  }

  _initializeTrainerState() {
    this.state = this.env.reset();
    this.metricsTracker.reset(this.agent);
    this.metrics = this.metricsTracker.data;
    this.episodeRewards = this.metricsTracker.episodeRewards;
    if (this.onStep) {
      this.onStep(this.state, 0, false, { ...this.metrics });
    }
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
      const weight = t.weight ?? 1;
      await this.agent.learn(t.state, t.action, t.reward, t.nextState, t.done, weight);
    }
  }

  _updateMetrics({ reward, done }) {
    this.metrics = this.metricsTracker.update(reward, this.agent);
    if (this.onStep) this.onStep(this.state, reward, done, { ...this.metrics });
  }

  _handleEpisodeEnd(done) {
    if (!done) return;
    this.metrics = this.metricsTracker.endEpisode(this.agent);
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

  _runLoop() {
    if (!this.isRunning) return;
    // use setTimeout so each step completes before scheduling the next
    this.timeout = setTimeout(async () => {
      if (this.isStepping) return;
      this.isStepping = true;
      try {
        await this.step();
      } catch (err) {
        console.error(err);
      } finally {
        this.isStepping = false;
        this._runLoop();
      }
    }, this.intervalMs);
  }

  start() {
    if (this.isRunning) return;
    this.state = this.env.reset();
    this.isRunning = true;
    this._runLoop();
  }

  setIntervalMs(ms) {
    this.intervalMs = ms;
    if (this.isRunning && !this.isStepping) {
      if (this.timeout) clearTimeout(this.timeout);
      this._runLoop();
    }
  }

  pause() {
    if (!this.isRunning) return;
    if (this.timeout) clearTimeout(this.timeout);
    this.isRunning = false;
  }

  reset() {
    this.pause();
    if (typeof this.agent.reset === 'function') {
      this.agent.reset();
    }
    this._initializeTrainerState();
  }

  resetTrainerState() {
    this.pause();
    this._initializeTrainerState();
  }

  async getAgentState() {
    if (!this.agent || typeof this.agent.toJSON !== 'function') {
      return null;
    }
    return this.agent.toJSON();
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
