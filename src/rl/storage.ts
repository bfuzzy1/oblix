import { RLAgent } from './agent.js';
import { RLTrainer } from './training.js';

export function saveAgent(
  agent: RLAgent,
  storage: Storage = globalThis.localStorage
): void {
  const data = JSON.stringify(agent.toJSON());
  storage.setItem('agent', data);
}

export function loadAgent(
  trainer: RLTrainer,
  storage: Storage = globalThis.localStorage
): RLAgent {
  const data = storage.getItem('agent');
  if (!data) return trainer.agent;
  const agent = RLAgent.fromJSON(JSON.parse(data));
  trainer.agent = agent;
  trainer.reset();
  return agent;
}
