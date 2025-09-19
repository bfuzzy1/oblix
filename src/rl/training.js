import { RLAgent } from './agent.js';
import { GridWorldEnvironment } from './environment.js';
import { ExperienceReplay } from './experienceReplay.js';
import { MetricsTracker } from './metricsTracker.js';

export class RLTrainer {
  constructor(agent, env, options = {}) {
    this.agent = agent;
    this.env = env;
    const configuredMaxSteps = Number(options.maxSteps ?? 50);
    this.maxSteps = Number.isFinite(configuredMaxSteps) && configuredMaxSteps > 0
      ? Math.trunc(configuredMaxSteps)
      : 50;
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
    this.stepsInEpisode = 0;
    this.metricsTracker = new MetricsTracker(this.agent);
    this.metrics = this.metricsTracker.data;
    this.episodeRewards = this.metricsTracker.episodeRewards;
    this._assignEnvironmentToAgent();
    this._initializeTrainerState();
  }

  _assignEnvironmentToAgent() {
    if (this.agent && typeof this.agent.setEnvironment === 'function') {
      this.agent.setEnvironment(this.env);
    }
  }

  _clearReplayBuffer() {
    if (this.replayBuffer && typeof this.replayBuffer.clear === 'function') {
      this.replayBuffer.clear();
    }
  }

  _emitStep(state, reward, done) {
    if (!this.onStep) return;
    this.onStep(state, reward, done, { ...this.metrics });
  }

  async _computeReplayTdError(sample) {
    const { state, action, reward, nextState, done } = sample;
    const weight = sample.weight ?? 1;
    let tdError = await this.agent.learn(state, action, reward, nextState, done, weight);
    if (Number.isFinite(tdError)) {
      return tdError;
    }

    if (typeof this.agent.computeTdError === 'function') {
      tdError = await this.agent.computeTdError(state, action, reward, nextState, done);
    } else if (typeof this.agent.tdError === 'function') {
      tdError = await this.agent.tdError(state, action, reward, nextState, done);
    }

    return Number.isFinite(tdError) ? tdError : null;
  }

  _resetInternal({ resetAgent }) {
    this.pause();
    this._assignEnvironmentToAgent();
    if (resetAgent && typeof this.agent.reset === 'function') {
      this.agent.reset(this.env);
    }
    this._clearReplayBuffer();
    this._initializeTrainerState();
  }

  _initializeTrainerState() {
    this.state = this.env.reset();
    this.stepsInEpisode = 0;
    this.metricsTracker.reset(this.agent);
    this.metrics = this.metricsTracker.data;
    this.episodeRewards = this.metricsTracker.episodeRewards;
    this._emitStep(this.state, 0, false);
  }

  async _applyTransition() {
    const action = await this.agent.act(this.state);
    const { state: nextState, reward, done } = this.env.step(action);
    return { state: this.state, action, reward, nextState, done };
  }

  async _processReplay(transition) {
    if (!this.replayBuffer) return;
    this.replayBuffer.add(transition, 1);
    const samples = this.replayBuffer.sample(this.replaySamples, this.replayStrategy);
    for (const sample of samples) {
      const tdError = await this._computeReplayTdError(sample);
      if (tdError !== null) {
        this.replayBuffer.updatePriority(sample.index, Math.abs(tdError));
      }
    }
  }

  _updateMetrics({ reward, done }) {
    this.metrics = this.metricsTracker.update(reward, this.agent);
    this._emitStep(this.state, reward, done);
  }

  _handleEpisodeEnd(done) {
    if (!done) return;
    this.metrics = this.metricsTracker.endEpisode(this.agent);
    this.state = this.env.reset();
    this.stepsInEpisode = 0;
    this._emitStep(this.state, 0, false);
  }

  async step() {
    if (!this.state) return;
    const transition = await this._applyTransition();
    this.stepsInEpisode += 1;
    const limitReached = Number.isFinite(this.maxSteps) && this.maxSteps > 0 && this.stepsInEpisode >= this.maxSteps;
    const done = transition.done || limitReached;
    await this.agent.learn(transition.state, transition.action, transition.reward, transition.nextState, done);
    this.state = transition.nextState;
    const updatedTransition = { ...transition, done };
    await this._processReplay(updatedTransition);
    this._updateMetrics(updatedTransition);
    this._handleEpisodeEnd(updatedTransition.done);
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
    this.stepsInEpisode = 0;
    
    if (!this.state) {
      this._initializeTrainerState();
    }
    
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

  setMaxSteps(limit) {
    const parsed = Number(limit);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return;
    }
    this.maxSteps = Math.trunc(parsed);
  }

  pause() {
    if (!this.isRunning) return;
    if (this.timeout) clearTimeout(this.timeout);
    this.isRunning = false;
  }

  reset() {
    this._resetInternal({ resetAgent: true });
  }

  resetTrainerState() {
    this._resetInternal({ resetAgent: false });
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
