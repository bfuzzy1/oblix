import { RLAgent } from './agent.js';
import { RLTrainer } from './training.js';

export function saveAgent(agent, storage = globalThis.localStorage) {
  const data = JSON.stringify(agent.toJSON());
  storage.setItem('agent', data);
}

export function loadAgent(trainer, storage = globalThis.localStorage) {
  const data = storage.getItem('agent');
  if (!data) return trainer.agent;
  const agent = RLAgent.fromJSON(JSON.parse(data));
  trainer.agent = agent;
  trainer.reset();
  return agent;
}
