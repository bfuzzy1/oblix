import { GridWorldEnvironment } from './environment.js';
import { RLTrainer } from './training.js';
import { createAgent } from '../ui/agentFactory.js';

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
  const envConfig = payload.env || {};
  const agentConfig = payload.agent || {};
  const trainerConfig = payload.trainer || {};
  const envSize = envConfig.size ?? environment?.size ?? 5;
  const obstacles = envConfig.obstacles || [];
  const rewardConfig = envConfig.rewards ? { ...envConfig.rewards } : null;
  const agentType = agentConfig.type || agent?.__factoryType || 'rl';
  const revision = agentConfig.revision;
  const intervalMs = trainerConfig.intervalMs ?? trainer?.intervalMs ?? 100;

  if (!trainer) {
    environment = new GridWorldEnvironment(envSize, obstacles, rewardConfig || undefined);
    agent = createAgent(agentType, agentConfig.params || {});
    trainer = new RLTrainer(agent, environment, {
      intervalMs,
      onStep: (state, reward, done, metrics) => {
        emitProgress(state, reward, done, metrics);
      }
    });
    trainer.resetTrainerState();
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

  const sizeChanged = !environment || environment.size !== envSize;
  if (!environment || sizeChanged) {
    environment = new GridWorldEnvironment(envSize, obstacles, rewardConfig || undefined);
  } else {
    if (typeof environment.setObstacles === 'function') {
      environment.setObstacles(obstacles);
    }
    if (rewardConfig && typeof environment.setRewardConfig === 'function') {
      environment.setRewardConfig(rewardConfig);
    }
  }

  const shouldRebuildTrainer = shouldReplaceAgent || sizeChanged;
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
  }
}
