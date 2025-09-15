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

function configureTrainer(payload) {
  if (!payload) return;
  if (trainer) {
    trainer.pause();
  }
  const envConfig = payload.env || {};
  const agentConfig = payload.agent || {};
  const trainerConfig = payload.trainer || {};
  environment = new GridWorldEnvironment(envConfig.size ?? 5, envConfig.obstacles || []);
  agent = createAgent(agentConfig.type || 'rl', agentConfig.params || {});
  trainer = new RLTrainer(agent, environment, {
    intervalMs: trainerConfig.intervalMs ?? 100,
    onStep: (state, reward, done, metrics) => {
      emitProgress(state, reward, done, metrics);
    }
  });
  trainer.resetTrainerState();
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
  }
}
