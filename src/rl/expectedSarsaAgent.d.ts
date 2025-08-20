import { RLAgent } from './agent.js';
export declare class ExpectedSarsaAgent extends RLAgent {
  learn(state: Float32Array, action: number, reward: number, nextState: Float32Array, done: boolean): void | Promise<void>;
}
