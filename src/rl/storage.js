import { RLAgent } from './agent.js';
import { RLTrainer } from './training.js';
import { DoubleQAgent } from './doubleQAgent.js';
import { OptimisticAgent } from './optimisticAgent.js';
import { ValueIterationAgent } from './valueIterationAgent.js';
import { PolicyIterationAgent } from './policyIterationAgent.js';

const AGENT_DESERIALIZERS = Object.freeze({
  double: DoubleQAgent.fromJSON,
  optimistic: OptimisticAgent.fromJSON,
  'value-iteration': ValueIterationAgent.fromJSON,
  'policy-iteration': PolicyIterationAgent.fromJSON
});

export function saveAgent(agent, storage = globalThis.localStorage) {
  const data = JSON.stringify(agent.toJSON());
  storage.setItem('agent', data);
}

export function loadAgent(trainer, storage = globalThis.localStorage) {
  const data = storage.getItem('agent');
  if (!data) return trainer.agent;
  const parsed = JSON.parse(data);
  const type = parsed.type ?? 'rl';
  const deserialize = AGENT_DESERIALIZERS[type];
  const agent = typeof deserialize === 'function'
    ? deserialize(parsed)
    : RLAgent.fromJSON(parsed);
  if (!agent || typeof agent !== 'object') {
    return trainer.agent;
  }
  Object.defineProperty(agent, '__factoryType', {
    value: type,
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

function cloneObstacles(obstacles = []) {
  return obstacles.map(obstacle => ({ x: obstacle.x, y: obstacle.y }));
}

function cloneScenarioConfig(env) {
  if (typeof env?.getScenarioConfig !== 'function') {
    return undefined;
  }
  const config = env.getScenarioConfig();
  if (config === undefined || config === null) {
    return config;
  }
  return JSON.parse(JSON.stringify(config));
}

export function saveEnvironment(env, storage = globalThis.localStorage) {
  const rewards = typeof env.getRewardConfig === 'function'
    ? env.getRewardConfig()
    : {
      stepPenalty: env.stepPenalty,
      obstaclePenalty: env.obstaclePenalty,
      goalReward: env.goalReward
    };
  const data = JSON.stringify({
    size: env.size,
    obstacles: cloneObstacles(env.obstacles),
    rewards,
    scenarioId: env.scenarioId ?? 'classic',
    scenarioConfig: cloneScenarioConfig(env)
  });
  storage.setItem('environment', data);
}

export function loadEnvironment(storage = globalThis.localStorage) {
  const data = storage.getItem('environment');
  if (!data) return null;
  const parsed = JSON.parse(data);
  const obstacles = Array.isArray(parsed.obstacles)
    ? parsed.obstacles.map(o => ({ x: o.x, y: o.y }))
    : [];
  return {
    size: parsed.size,
    obstacles,
    rewards: parsed.rewards,
    scenarioId: parsed.scenarioId ?? 'classic',
    scenarioConfig: parsed.scenarioConfig
  };
}
