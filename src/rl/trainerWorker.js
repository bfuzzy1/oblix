import { DEFAULT_REWARD_CONFIG } from './environment.js';
import { RLTrainer } from './training.js';
import { createAgent } from '../ui/agentFactory.js';
import { createEnvironmentFromScenario, DEFAULT_SCENARIO_ID } from './environmentPresets.js';

let postToHost = null;

const canUsePostMessage = typeof globalThis !== 'undefined'
  && typeof globalThis.postMessage === 'function'
  && typeof globalThis.addEventListener === 'function';

if (canUsePostMessage) {
  postToHost = data => globalThis.postMessage(data);
  globalThis.addEventListener('message', event => {
    handleMessage(event.data);
  });
} else {
  let parentPort = null;
  try {
    ({ parentPort } = await import('node:worker_threads'));
  } catch (err) {
    parentPort = null;
  }
  if (!parentPort) {
    throw new Error('trainerWorker: no messaging interface available');
  }
  postToHost = data => parentPort.postMessage(data);
  parentPort.on('message', message => {
    handleMessage(message);
  });
}

let trainer = null;
let agent = null;
let environment = null;
let agentRevision = null;
let lastEnvConfig = null;

function cloneObstacles(obstacles = []) {
  if (!Array.isArray(obstacles)) return [];
  return obstacles.map(obstacle => ({
    x: obstacle?.x,
    y: obstacle?.y
  }));
}

function sanitizeReward(value, fallback) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function cloneRewardConfig(rewards) {
  if (!rewards || typeof rewards !== 'object') return undefined;
  return {
    stepPenalty: sanitizeReward(rewards.stepPenalty, DEFAULT_REWARD_CONFIG.stepPenalty),
    obstaclePenalty: sanitizeReward(rewards.obstaclePenalty, DEFAULT_REWARD_CONFIG.obstaclePenalty),
    goalReward: sanitizeReward(rewards.goalReward, DEFAULT_REWARD_CONFIG.goalReward)
  };
}

function cloneScenarioConfig(config) {
  if (config === undefined || config === null) {
    return undefined;
  }
  return JSON.parse(JSON.stringify(config));
}

function areObstaclesEqual(a, b) {
  const listA = Array.isArray(a) ? a : [];
  const listB = Array.isArray(b) ? b : [];
  if (listA.length !== listB.length) return false;
  for (let i = 0; i < listA.length; i += 1) {
    const left = listA[i];
    const right = listB[i];
    if (!left && !right) continue;
    if (!left || !right) return false;
    if (left.x !== right.x || left.y !== right.y) {
      return false;
    }
  }
  return true;
}

function areRewardConfigsEqual(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.stepPenalty === b.stepPenalty
    && a.obstaclePenalty === b.obstaclePenalty
    && a.goalReward === b.goalReward;
}

function areScenarioConfigsEqual(a, b) {
  const normalizedA = a === null ? undefined : a;
  const normalizedB = b === null ? undefined : b;
  if (normalizedA === undefined && normalizedB === undefined) {
    return true;
  }
  if (normalizedA === undefined || normalizedB === undefined) {
    return false;
  }
  return JSON.stringify(normalizedA) === JSON.stringify(normalizedB);
}

function applyScenarioConfig(environmentInstance, scenarioConfig) {
  if (!environmentInstance || scenarioConfig === undefined) {
    return false;
  }
  if (typeof environmentInstance.applyScenarioOptions === 'function') {
    environmentInstance.applyScenarioOptions(scenarioConfig);
    return true;
  }
  if (typeof environmentInstance.setScenarioConfig === 'function') {
    environmentInstance.setScenarioConfig(scenarioConfig);
    return true;
  }
  if (environmentInstance.scenarioId === 'windy' && typeof environmentInstance.setWind === 'function') {
    const config = scenarioConfig?.windColumns ? scenarioConfig.windColumns : [];
    environmentInstance.setWind(config);
    return true;
  }
  if (environmentInstance.scenarioId === 'reward-grid' && typeof environmentInstance.setRewardCells === 'function') {
    const cells = Array.isArray(scenarioConfig?.rewardCells) ? scenarioConfig.rewardCells : [];
    environmentInstance.setRewardCells(cells);
    return true;
  }
  return false;
}

function snapshotEnvironmentConfig(env) {
  if (!env) return null;
  const rewards = typeof env.getRewardConfig === 'function'
    ? cloneRewardConfig(env.getRewardConfig())
    : undefined;
  const scenarioConfig = typeof env.getScenarioConfig === 'function'
    ? cloneScenarioConfig(env.getScenarioConfig())
    : undefined;
  return {
    scenarioId: env.scenarioId ?? DEFAULT_SCENARIO_ID,
    size: env.size,
    obstacles: cloneObstacles(env.obstacles),
    rewards,
    scenarioConfig
  };
}

function createEnvironmentInstance(config) {
  const options = {
    size: config.size,
    obstacles: cloneObstacles(config.obstacles)
  };
  if (config.rewards) {
    options.rewards = { ...config.rewards };
  }
  if (config.scenarioConfig !== undefined) {
    options.scenarioConfig = cloneScenarioConfig(config.scenarioConfig);
  }
  const id = config.scenarioId || DEFAULT_SCENARIO_ID;
  return createEnvironmentFromScenario(id, options);
}

function emitEnvironmentMetadata(requestId) {
  if (!postToHost) return;
  if (!environment) {
    postToHost({
      type: 'env:metadata',
      payload: { requestId, metadata: null }
    });
    return;
  }
  const rewards = typeof environment.getRewardConfig === 'function'
    ? cloneRewardConfig(environment.getRewardConfig())
    : undefined;
  const scenarioConfig = typeof environment.getScenarioConfig === 'function'
    ? cloneScenarioConfig(environment.getScenarioConfig())
    : undefined;
  postToHost({
    type: 'env:metadata',
    payload: {
      requestId,
      metadata: {
        scenarioId: environment.scenarioId ?? DEFAULT_SCENARIO_ID,
        size: environment.size,
        obstacles: cloneObstacles(environment.obstacles),
        rewards,
        scenarioConfig
      }
    }
  });
}

function emitProgress(state, reward, done, metrics) {
  if (!trainer || !postToHost) return;
  postToHost({
    type: 'progress',
    payload: {
      state,
      reward,
      done,
      metrics,
      episodeRewards: trainer.episodeRewards.slice()
    }
  });
}

function emitAgentState(requestId) {
  if (!postToHost) return;
  const snapshot = agent && typeof agent.toJSON === 'function'
    ? agent.toJSON()
    : null;
  postToHost({
    type: 'agent:state',
    payload: {
      requestId,
      agent: snapshot
    }
  });
}

function configureTrainer(payload) {
  if (!payload) return;
  const envConfigPayload = payload.env || {};
  const agentConfig = payload.agent || {};
  const trainerConfig = payload.trainer || {};

  const scenarioId = typeof envConfigPayload.scenarioId === 'string'
    ? envConfigPayload.scenarioId
    : lastEnvConfig?.scenarioId ?? environment?.scenarioId ?? DEFAULT_SCENARIO_ID;
  const size = Number.isFinite(envConfigPayload.size)
    ? Math.trunc(envConfigPayload.size)
    : lastEnvConfig?.size ?? environment?.size ?? 5;
  const obstacles = Array.isArray(envConfigPayload.obstacles)
    ? cloneObstacles(envConfigPayload.obstacles)
    : cloneObstacles(lastEnvConfig?.obstacles ?? environment?.obstacles ?? []);
  const rewards = envConfigPayload.rewards
    ? cloneRewardConfig(envConfigPayload.rewards)
    : (lastEnvConfig?.rewards ? { ...lastEnvConfig.rewards } : undefined);
  const hasScenarioConfig = Object.prototype.hasOwnProperty.call(envConfigPayload, 'scenarioConfig');
  const scenarioConfig = hasScenarioConfig
    ? cloneScenarioConfig(envConfigPayload.scenarioConfig)
    : cloneScenarioConfig(lastEnvConfig?.scenarioConfig);

  const agentType = agentConfig.type || agent?.__factoryType || 'rl';
  const revision = agentConfig.revision;
  const intervalMs = trainerConfig.intervalMs ?? trainer?.intervalMs ?? 100;

  const normalizedEnvConfig = { scenarioId, size, obstacles, rewards, scenarioConfig };

  if (!trainer) {
    environment = createEnvironmentInstance(normalizedEnvConfig);
    agent = createAgent(agentType, agentConfig.params || {});
    trainer = new RLTrainer(agent, environment, {
      intervalMs,
      onStep: (state, reward, done, metrics) => {
        emitProgress(state, reward, done, metrics);
      }
    });
    trainer.resetTrainerState();
    lastEnvConfig = snapshotEnvironmentConfig(environment);
    if (revision !== undefined) {
      agentRevision = revision;
    }
    return;
  }

  const wasRunning = trainer.isRunning;
  const typeChanged = agent?.__factoryType !== agentType;
  const revisionChanged = revision !== undefined && revision !== agentRevision;
  const shouldReplaceAgent = !agent || typeChanged || revisionChanged;

  if (shouldReplaceAgent) {
    agent = createAgent(agentType, agentConfig.params || {});
  }

  const scenarioChanged = !lastEnvConfig || lastEnvConfig.scenarioId !== scenarioId;
  const sizeChanged = !lastEnvConfig || lastEnvConfig.size !== size;
  const obstaclesChanged = !lastEnvConfig || !areObstaclesEqual(lastEnvConfig.obstacles, obstacles);
  const rewardsChanged = !lastEnvConfig || !areRewardConfigsEqual(lastEnvConfig.rewards, rewards);
  const scenarioConfigChanged = !lastEnvConfig || !areScenarioConfigsEqual(lastEnvConfig.scenarioConfig, scenarioConfig);

  let shouldRebuildEnvironment = !environment || scenarioChanged || sizeChanged;

  if (!shouldRebuildEnvironment && scenarioConfigChanged) {
    const scenarioClone = cloneScenarioConfig(scenarioConfig);
    const applied = applyScenarioConfig(environment, scenarioClone);
    if (!applied) {
      shouldRebuildEnvironment = true;
    }
  }

  if (!shouldRebuildEnvironment && obstaclesChanged && typeof environment.setObstacles === 'function') {
    environment.setObstacles(obstacles);
  }

  if (!shouldRebuildEnvironment && rewardsChanged && typeof environment.setRewardConfig === 'function') {
    environment.setRewardConfig(rewards || {});
  }

  if (shouldRebuildEnvironment) {
    environment = createEnvironmentInstance(normalizedEnvConfig);
  }

  lastEnvConfig = snapshotEnvironmentConfig(environment);

  const shouldRebuildTrainer = shouldReplaceAgent || shouldRebuildEnvironment;
  if (shouldRebuildTrainer && trainer) {
    trainer.pause();
  }

  if (shouldRebuildTrainer) {
    trainer = new RLTrainer(agent, environment, {
      intervalMs,
      onStep: (state, reward, done, metrics) => {
        emitProgress(state, reward, done, metrics);
      }
    });
    trainer.resetTrainerState();
  } else {
    const intervalChanged = typeof trainerConfig.intervalMs === 'number'
      && trainerConfig.intervalMs !== trainer.intervalMs;
    if (intervalChanged) {
      trainer.setIntervalMs(trainerConfig.intervalMs);
    }
    trainer.env = environment;
    if (!trainer.state) {
      trainer.resetTrainerState();
    }
  }

  if (revision !== undefined) {
    agentRevision = revision;
  }

  if (wasRunning && shouldRebuildTrainer) {
    trainer.start();
  }
}

function updateAgent(params) {
  if (!agent || !params) return;
  Object.assign(agent, params);
  if (trainer) {
    trainer.metrics.epsilon = agent.epsilon;
    emitProgress(trainer.state, 0, false, { ...trainer.metrics });
  }
}

function handleMessage(message) {
  if (!message || typeof message !== 'object') return;
  const { type, payload } = message;
  switch (type) {
    case 'config':
      configureTrainer(payload);
      break;
    case 'start':
      trainer?.start();
      break;
    case 'pause':
      trainer?.pause();
      break;
    case 'reset':
      trainer?.reset();
      break;
    case 'resetTrainerState':
      trainer?.resetTrainerState();
      break;
    case 'interval':
      if (trainer && typeof payload === 'number') {
        trainer.setIntervalMs(payload);
      }
      break;
    case 'agent:update':
      updateAgent(payload);
      break;
    case 'agent:getState':
      emitAgentState(payload?.requestId);
      break;
    case 'env:getMetadata':
      emitEnvironmentMetadata(payload?.requestId);
      break;
  }
}
