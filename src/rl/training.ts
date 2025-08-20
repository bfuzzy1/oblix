import { RLAgent } from './agent.js';
import {
  GridWorldEnvironment,
  EnvironmentState
} from './environment.js';

export interface TrainingMetrics {
  episode: number;
  steps: number;
  cumulativeReward: number;
  epsilon: number;
}

export interface TrainerOptions {
  maxSteps?: number;
  intervalMs?: number;
  liveChart?: { push: (reward: number, epsilon: number) => void } | null;
  onStep?: (
    state: EnvironmentState,
    reward: number,
    done: boolean,
    metrics: TrainingMetrics
  ) => void;
}

export class RLTrainer {
  agent: RLAgent;
  env: GridWorldEnvironment;
  maxSteps: number;
  intervalMs: number;
  liveChart: { push: (reward: number, epsilon: number) => void } | null;
  onStep: (
    state: EnvironmentState,
    reward: number,
    done: boolean,
    metrics: TrainingMetrics
  ) => void;
  isRunning: boolean;
  interval: ReturnType<typeof setInterval> | null;
  state: EnvironmentState | null;
  metrics: TrainingMetrics;
  episodeRewards: number[];

  constructor(agent: RLAgent, env: GridWorldEnvironment, options: TrainerOptions = {}) {
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

  async step(): Promise<void> {
    if (!this.state) return;
    const action = this.agent.act(this.state);
    const { state: nextState, reward, done } = this.env.step(action);
    await this.agent.learn(this.state, action, reward, nextState, done);
    this.state = nextState;
    this.metrics.steps += 1;
    this.metrics.cumulativeReward += reward;
    this.metrics.epsilon = this.agent.epsilon;
    if (this.onStep) this.onStep(this.state, reward, done, { ...this.metrics });
    if (done) {
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
  }

  start(): void {
    if (this.isRunning) return;
    this.state = this.env.reset();
    this.isRunning = true;
    this.interval = setInterval(async () => {
      await this.step();
    }, this.intervalMs);
  }

  setIntervalMs(ms: number): void {
    this.intervalMs = ms;
    if (this.isRunning) {
      if (this.interval) clearInterval(this.interval);
      this.interval = setInterval(async () => {
        await this.step();
      }, this.intervalMs);
    }
  }

  pause(): void {
    if (!this.isRunning) return;
    if (this.interval) clearInterval(this.interval);
    this.isRunning = false;
  }

  reset(): void {
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

  static async trainEpisodes(
    agent: RLAgent,
    env: GridWorldEnvironment,
    episodes = 10,
    maxSteps = 50
  ): Promise<void> {
    for (let ep = 0; ep < episodes; ep++) {
      let state = env.reset();
      for (let st = 0; st < maxSteps; st++) {
        const action = agent.act(state);
        const { state: nextState, reward, done } = env.step(action);
        await agent.learn(state, action, reward, nextState, done);
        state = nextState;
        if (done) break;
      }
    }
  }
}
