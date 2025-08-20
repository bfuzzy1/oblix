import { RLAgent, AgentOptions } from './agent.js';
interface DynaQOptions extends AgentOptions {
  planningSteps?: number;
}
interface ModelEntry {
  nextState: Float32Array;
  reward: number;
  done: boolean;
}
export declare class DynaQAgent extends RLAgent {
  planningSteps: number;
  model: Map<string, ModelEntry>;
  stateActions: {
  state: Float32Array;
  action: number;
  }[];
  constructor(options?: DynaQOptions);
  private _saKey;
  learn(state: Float32Array, action: number, reward: number, nextState: Float32Array, done: boolean): void | Promise<void>;
}
export {};
