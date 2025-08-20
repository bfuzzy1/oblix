import { RLAgent } from './agent.js';
import { RLTrainer } from './training.js';
export declare function saveAgent(agent: RLAgent, storage?: Storage): void;
export declare function loadAgent(trainer: RLTrainer, storage?: Storage): RLAgent;
