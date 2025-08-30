import { RLAgent } from './agent.js';
import { RLTrainer } from './training.js';
import { DoubleQAgent } from './doubleQAgent.js';
import { OptimisticAgent } from './optimisticAgent.js';

export function saveAgent(agent, storage = globalThis.localStorage) {
  const data = JSON.stringify(agent.toJSON());
  storage.setItem('agent', data);
}

export function loadAgent(trainer, storage = globalThis.localStorage) {
  const data = storage.getItem('agent');
  if (!data) return trainer.agent;
  const parsed = JSON.parse(data);
  let agent;
  if (parsed.type === 'double') {
    agent = DoubleQAgent.fromJSON(parsed);
  } else if (parsed.type === 'optimistic') {
    agent = OptimisticAgent.fromJSON(parsed);
  } else {
    agent = RLAgent.fromJSON(parsed);
  }
  trainer.agent = agent;
  trainer.reset();
  return agent;
}
