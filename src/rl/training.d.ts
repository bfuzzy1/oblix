import { RLAgent } from './agent.js';
import { GridWorldEnvironment, EnvironmentState } from './environment.js';
export interface TrainingMetrics {
  episode: number;
  steps: number;
  cumulativeReward: number;
  epsilon: number;
}
export interface TrainerOptions {
  maxSteps?: number;
  intervalMs?: number;
  liveChart?: {
  push: (reward: number, epsilon: number) => void;
  } | null;
  onStep?: (state: EnvironmentState, reward: number, done: boolean, metrics: TrainingMetrics) => void;
}
export declare class RLTrainer {
  agent: RLAgent;
  env: GridWorldEnvironment;
  maxSteps: number;
  intervalMs: number;
  liveChart: {
  push: (reward: number, epsilon: number) => void;
  } | null;
  onStep: (state: EnvironmentState, reward: number, done: boolean, metrics: TrainingMetrics) => void;
  isRunning: boolean;
  interval: ReturnType<typeof setInterval> | null;
  state: EnvironmentState | null;
  metrics: TrainingMetrics;
  episodeRewards: number[];
  constructor(agent: RLAgent, env: GridWorldEnvironment, options?: TrainerOptions);
  step(): Promise<void>;
  start(): void;
  setIntervalMs(ms: number): void;
  pause(): void;
  reset(): void;
  static trainEpisodes(agent: RLAgent, env: GridWorldEnvironment, episodes?: number, maxSteps?: number): Promise<void>;
}
