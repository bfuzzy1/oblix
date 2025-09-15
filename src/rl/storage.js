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
  Object.defineProperty(agent, '__factoryType', {
    value: parsed.type || 'rl',
    writable: true,
    configurable: true,
    enumerable: false
  });
  let assigned = agent;
  if (typeof trainer.setAgent === 'function') {
    const result = trainer.setAgent(agent);
    assigned = result || trainer.agent;
  } else {
    trainer.agent = agent;
  }
  if (typeof trainer.resetTrainerState === 'function') {
    trainer.resetTrainerState();
  } else {
    trainer.reset();
  }
  return assigned;
}

export function saveEnvironment(env, storage = globalThis.localStorage) {
  const data = JSON.stringify({ size: env.size, obstacles: env.obstacles });
  storage.setItem('environment', data);
}

export function loadEnvironment(storage = globalThis.localStorage) {
  const data = storage.getItem('environment');
  if (!data) return null;
  return JSON.parse(data);
}
