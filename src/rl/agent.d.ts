export interface AgentOptions {
  epsilon?: number;
  gamma?: number;
  learningRate?: number;
  epsilonDecay?: number;
  minEpsilon?: number;
}
export declare class RLAgent {
  initialEpsilon: number;
  epsilon: number;
  gamma: number;
  learningRate: number;
  epsilonDecay: number;
  minEpsilon: number;
  qTable: Map<string, Float32Array>;
  constructor(options?: AgentOptions);
  protected _key(state: Float32Array): string;
  protected _ensure(state: Float32Array): Float32Array;
  /** Choose an action using epsilon-greedy policy. */
  act(state: Float32Array): number;
  /** Reduce exploration rate after learning. */
  decayEpsilon(): void;
  /** Perform tabular Q-learning update. */
  learn(state: Float32Array, action: number, reward: number, nextState: Float32Array, done: boolean): void | Promise<void>;
  /** Reset agent to initial state. */
  reset(): void;
  /** Serialize agent state to a plain object. */
  toJSON(): Record<string, unknown>;
  /** Recreate an agent from serialized data. */
  static fromJSON(data: any): RLAgent;
}
