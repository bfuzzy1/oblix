import { DEFAULT_REWARD_CONFIG } from '../rl/environment.js';
import { LiveChart } from './liveChart.js';
import { createAgent } from './agentFactory.js';
import { initRenderer, render } from './renderGrid.js';
import { bindControls } from './bindControls.js';
import { saveEnvironment } from '../rl/storage.js';
import { createEnvironmentFromScenario, getScenarioDefinitions, DEFAULT_SCENARIO_ID } from '../rl/environmentPresets.js';

const supportsWorker = typeof Worker !== 'undefined';

const gridEl = document.getElementById('grid');
const gridSizeInput = document.getElementById('grid-size');
const stepPenaltyInput = document.getElementById('step-penalty');
const obstaclePenaltyInput = document.getElementById('obstacle-penalty');
const goalRewardInput = document.getElementById('goal-reward');
const scenarioSelect = document.getElementById('scenario-select');

const scenarioDefinitions = getScenarioDefinitions();
if (scenarioSelect) {
  scenarioSelect.innerHTML = '';
  for (const scenario of scenarioDefinitions) {
    const option = document.createElement('option');
    option.value = scenario.id;
    option.textContent = scenario.label;
    scenarioSelect.appendChild(option);
  }
  scenarioSelect.value = DEFAULT_SCENARIO_ID;
}

function parseGridSize(value) {
  const size = parseInt(value, 10);
  return Number.isFinite(size) ? size : 5;
}

function parseRewardValue(inputEl, fallback) {
  if (!inputEl) return fallback;
  const num = Number(inputEl.value);
  return Number.isFinite(num) ? num : fallback;
}

function getRewardConfigFromInputs() {
  return {
    stepPenalty: parseRewardValue(stepPenaltyInput, DEFAULT_REWARD_CONFIG.stepPenalty),
    obstaclePenalty: parseRewardValue(obstaclePenaltyInput, DEFAULT_REWARD_CONFIG.obstaclePenalty),
    goalReward: parseRewardValue(goalRewardInput, DEFAULT_REWARD_CONFIG.goalReward)
  };
}

function syncRewardInputsFromEnv(environment) {
  if (!environment) return;
  const rewards = typeof environment.getRewardConfig === 'function'
    ? environment.getRewardConfig()
    : {
      stepPenalty: environment.stepPenalty,
      obstaclePenalty: environment.obstaclePenalty,
      goalReward: environment.goalReward
    };
  if (stepPenaltyInput) stepPenaltyInput.value = rewards.stepPenalty;
  if (obstaclePenaltyInput) obstaclePenaltyInput.value = rewards.obstaclePenalty;
  if (goalRewardInput) goalRewardInput.value = rewards.goalReward;
}

function cloneObstacles(obstacles = []) {
  return obstacles.map(obstacle => ({ x: obstacle.x, y: obstacle.y }));
}

function cloneScenarioConfig(environment) {
  if (!environment || typeof environment.getScenarioConfig !== 'function') {
    return undefined;
  }
  const config = environment.getScenarioConfig();
  if (config === undefined || config === null) {
    return config;
  }
  return JSON.parse(JSON.stringify(config));
}

const initialScenarioId = scenarioSelect?.value || DEFAULT_SCENARIO_ID;
let currentScenarioId = initialScenarioId;
let env = createEnvironmentFromScenario(initialScenarioId, {
  size: parseGridSize(gridSizeInput.value),
  rewards: getRewardConfigFromInputs()
});
currentScenarioId = env.scenarioId ?? initialScenarioId;
if (scenarioSelect) {
  scenarioSelect.value = currentScenarioId;
}
gridSizeInput.value = env.size;
syncRewardInputsFromEnv(env);
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

function rebuildEnvironment(config = {}) {
  const useDefaults = Boolean(config.useDefaults);
  const scenarioId = config.scenarioId ?? currentScenarioId;
  const options = {};
  if (config.size !== undefined) {
    options.size = config.size;
  } else if (!useDefaults && env) {
    options.size = env.size;
  } else if (!useDefaults) {
    options.size = parseGridSize(gridSizeInput.value);
  }
  if (config.obstacles !== undefined) {
    options.obstacles = cloneObstacles(config.obstacles);
  } else if (!useDefaults && env) {
    options.obstacles = cloneObstacles(env.obstacles);
  }
  if (config.rewards !== undefined) {
    options.rewards = config.rewards;
  } else if (!useDefaults) {
    options.rewards = getRewardConfigFromInputs();
  }
  if (config.scenarioConfig !== undefined) {
    options.scenarioConfig = config.scenarioConfig;
  } else if (!useDefaults) {
    options.scenarioConfig = cloneScenarioConfig(env);
  }
  const nextEnv = createEnvironmentFromScenario(scenarioId, options);
  env = nextEnv;
  currentScenarioId = nextEnv.scenarioId ?? scenarioId;
  if (scenarioSelect) {
    scenarioSelect.value = currentScenarioId;
  }
  gridSizeInput.value = env.size;
  syncRewardInputsFromEnv(env);
  initRenderer(env, gridEl, env.size, handleEnvironmentChange);
  if (trainer) {
    if (typeof trainer.setEnvironment === 'function') {
      trainer.setEnvironment(env);
    } else {
      trainer.env = env;
    }
    trainer.reset();
  }
  saveEnvironment(env);
  const state = typeof env.getState === 'function' ? env.getState() : null;
  if (state) {
    render(state);
  }
  return env;
}

gridSizeInput.addEventListener('change', e => {
  const newSize = parseGridSize(e.target.value);
  rebuildEnvironment({ size: newSize });
});

bindControls(trainer, agent, render, () => env, rebuildEnvironment);

saveEnvironment(env);
render(env.getState());

function createWorkerTrainer(initialAgent, initialEnv, options) {
  const worker = new Worker(new URL('../rl/trainerWorker.js', import.meta.url), { type: 'module' });
  let currentEnv = initialEnv;
  let rawAgent = initialAgent;
  let agentRevision = 0;
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
      agentRevision += 1;
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

  function getEnvRewardConfig(environment) {
    if (!environment) return null;
    if (typeof environment.getRewardConfig === 'function') {
      return environment.getRewardConfig();
    }
    if (
      environment.stepPenalty !== undefined
      || environment.obstaclePenalty !== undefined
      || environment.goalReward !== undefined
    ) {
      return {
        stepPenalty: environment.stepPenalty,
        obstaclePenalty: environment.obstaclePenalty,
        goalReward: environment.goalReward
      };
    }
    return null;
  }

  function sanitizeReward(value, fallback) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  }

  function cloneRewardConfig(rewards) {
    if (!rewards) return undefined;
    return {
      stepPenalty: sanitizeReward(rewards.stepPenalty, DEFAULT_REWARD_CONFIG.stepPenalty),
      obstaclePenalty: sanitizeReward(rewards.obstaclePenalty, DEFAULT_REWARD_CONFIG.obstaclePenalty),
      goalReward: sanitizeReward(rewards.goalReward, DEFAULT_REWARD_CONFIG.goalReward)
    };
  }

  function sendConfig() {
    const rewards = cloneRewardConfig(getEnvRewardConfig(currentEnv));
    worker.postMessage({
      type: 'config',
      payload: {
        agent: {
          type: rawAgent.__factoryType || 'rl',
          params: serializeAgent(rawAgent),
          revision: agentRevision
        },
        env: {
          size: currentEnv.size,
          obstacles: cloneObstacles(currentEnv.obstacles),
          ...(rewards ? { rewards } : {})
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
