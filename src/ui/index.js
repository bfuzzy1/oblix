import { GridWorldEnvironment } from '../rl/environment.js';
import { LiveChart } from './liveChart.js';
import { createAgent } from './agentFactory.js';
import { initRenderer, render } from './renderGrid.js';
import { bindControls } from './bindControls.js';
import { saveEnvironment } from '../rl/storage.js';

const supportsWorker = typeof Worker !== 'undefined';

const gridEl = document.getElementById('grid');
const gridSizeInput = document.getElementById('grid-size');
let env = new GridWorldEnvironment(parseInt(gridSizeInput.value, 10));
let trainer;
let agent;

function handleEnvironmentChange(updatedEnv = env) {
  if (!trainer) return;
  const nextEnv = updatedEnv || env;
  if (typeof trainer.setEnvironment === 'function') {
    trainer.setEnvironment(nextEnv);
  }
}

initRenderer(env, gridEl, env.size, handleEnvironmentChange);

const policySelect = document.getElementById('policy-select');
const lambdaSlider = document.getElementById('lambda-slider');

let baseAgent = createAgent('rl', {
  policy: policySelect.value,
  lambda: parseFloat(lambdaSlider.value)
});

const liveChart = new LiveChart(document.getElementById('liveChart'));
const metricsEls = {
  episode: document.getElementById('episode'),
  steps: document.getElementById('steps'),
  reward: document.getElementById('reward'),
  epsilon: document.getElementById('epsilon'),
  epsilonSlider: document.getElementById('epsilon-slider'),
  epsilonValue: document.getElementById('epsilon-value')
};

function handleProgress(state, reward, done, metrics) {
  render(state);
  metricsEls.episode.textContent = metrics.episode;
  metricsEls.steps.textContent = metrics.steps;
  metricsEls.reward.textContent = metrics.cumulativeReward.toFixed(2);
  metricsEls.epsilon.textContent = metrics.epsilon.toFixed(2);
  metricsEls.epsilonSlider.value = metrics.epsilon;
  metricsEls.epsilonValue.textContent = metrics.epsilon.toFixed(2);
}

if (supportsWorker) {
  trainer = createWorkerTrainer(baseAgent, env, {
    intervalMs: 100,
    liveChart,
    onProgress: handleProgress
  });
  agent = trainer.agent;
} else {
  const { RLTrainer } = await import('../rl/training.js');
  trainer = new RLTrainer(baseAgent, env, {
    intervalMs: 100,
    liveChart,
    onStep: handleProgress
  });
  agent = baseAgent;
}

function rebuildEnvironment(size, obstacles = []) {
  env = new GridWorldEnvironment(size, obstacles);
  if (typeof trainer.setEnvironment === 'function') {
    trainer.setEnvironment(env);
  } else {
    trainer.env = env;
  }
  initRenderer(env, gridEl, size, handleEnvironmentChange);
  trainer.reset();
  gridSizeInput.value = size;
  saveEnvironment(env);
}

gridSizeInput.addEventListener('change', e => {
  const newSize = parseInt(e.target.value, 10);
  rebuildEnvironment(newSize);
});

bindControls(trainer, agent, render, () => env, rebuildEnvironment);

render(env.reset());

function createWorkerTrainer(initialAgent, initialEnv, options) {
  const worker = new Worker(new URL('../rl/trainerWorker.js', import.meta.url), { type: 'module' });
  let currentEnv = initialEnv;
  let rawAgent = initialAgent;
  const metrics = {
    episode: 1,
    steps: 0,
    cumulativeReward: 0,
    epsilon: initialAgent.epsilon ?? 1
  };
  const pendingAgentStateRequests = new Map();
  let nextAgentStateRequestId = 0;

  function fallbackAgentState() {
    if (!rawAgent || typeof rawAgent.toJSON !== 'function') {
      return null;
    }
    return rawAgent.toJSON();
  }

  function requestAgentState() {
    if (!worker) {
      return Promise.resolve(fallbackAgentState());
    }
    return new Promise(resolve => {
      const requestId = nextAgentStateRequestId++;
      const timeout = setTimeout(() => {
        pendingAgentStateRequests.delete(requestId);
        resolve(fallbackAgentState());
      }, 1000);
      pendingAgentStateRequests.set(requestId, { resolve, timeout });
      worker.postMessage({
        type: 'agent:getState',
        payload: { requestId }
      });
    });
  }

  const trainerProxy = {
    intervalMs: options.intervalMs ?? 100,
    metrics,
    state: typeof currentEnv.getState === 'function' ? currentEnv.getState() : null,
    episodeRewards: [],
    agent: null,
    start() {
      worker.postMessage({ type: 'start' });
    },
    pause() {
      worker.postMessage({ type: 'pause' });
    },
    reset() {
      worker.postMessage({ type: 'reset' });
    },
    resetTrainerState() {
      worker.postMessage({ type: 'resetTrainerState' });
    },
    setIntervalMs(ms) {
      this.intervalMs = ms;
      worker.postMessage({ type: 'interval', payload: ms });
    },
    setAgent(newAgent) {
      rawAgent = newAgent;
      const proxy = wrapAgent(newAgent);
      this.agent = proxy;
      metrics.epsilon = newAgent.epsilon ?? metrics.epsilon;
      sendConfig();
      return proxy;
    },
    setEnvironment(newEnv) {
      currentEnv = newEnv;
      if (typeof newEnv.getState === 'function') {
        this.state = newEnv.getState();
      }
      sendConfig();
    },
    getAgentState: requestAgentState
  };

  function wrapAgent(agentInstance) {
    return new Proxy(agentInstance, {
      set(target, prop, value) {
        target[prop] = value;
        if (prop !== '__factoryType') {
          worker.postMessage({
            type: 'agent:update',
            payload: { [prop]: value }
          });
        }
        return true;
      },
      get(target, prop) {
        return target[prop];
      }
    });
  }

  function serializeAgent(agentInstance) {
    const fields = [
      'epsilon',
      'epsilonDecay',
      'minEpsilon',
      'policy',
      'learningRate',
      'lambda',
      'gamma',
      'temperature',
      'ucbC',
      'alphaCritic',
      'alphaActor',
      'alpha',
      'beta',
      'planningSteps',
      'exploringStarts',
      'initialValue'
    ];
    const params = {};
    for (const key of fields) {
      if (agentInstance[key] !== undefined) {
        params[key] = agentInstance[key];
      }
    }
    return params;
  }

  function cloneObstacles(obstacles) {
    return (obstacles || []).map(o => ({ x: o.x, y: o.y }));
  }

  function sendConfig() {
    worker.postMessage({
      type: 'config',
      payload: {
        agent: {
          type: rawAgent.__factoryType || 'rl',
          params: serializeAgent(rawAgent)
        },
        env: {
          size: currentEnv.size,
          obstacles: cloneObstacles(currentEnv.obstacles)
        },
        trainer: {
          intervalMs: trainerProxy.intervalMs
        }
      }
    });
  }

  trainerProxy.agent = wrapAgent(rawAgent);
  metrics.epsilon = rawAgent.epsilon ?? metrics.epsilon;

  worker.onmessage = event => {
    const { type, payload } = event.data || {};
    if (type === 'agent:state') {
      const requestId = payload?.requestId;
      if (requestId !== undefined && pendingAgentStateRequests.has(requestId)) {
        const { resolve, timeout } = pendingAgentStateRequests.get(requestId);
        pendingAgentStateRequests.delete(requestId);
        clearTimeout(timeout);
        resolve(payload?.agent ?? fallbackAgentState());
      }
      return;
    }
    if (type !== 'progress' || !payload) return;
    trainerProxy.state = payload.state;
    trainerProxy.episodeRewards = payload.episodeRewards || trainerProxy.episodeRewards;
    if (payload.metrics) {
      Object.assign(trainerProxy.metrics, payload.metrics);
    }
    if (typeof options.onProgress === 'function' && payload.metrics) {
      options.onProgress(payload.state, payload.reward, payload.done, payload.metrics);
    }
    if (options.liveChart && payload.metrics) {
      options.liveChart.push(payload.metrics.cumulativeReward, payload.metrics.epsilon);
    }
  };

  sendConfig();

  return trainerProxy;
}
