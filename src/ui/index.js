import { DEFAULT_REWARD_CONFIG } from '../rl/environment.js';
import { LiveChart } from './liveChart.js';
import { createAgent } from './agentFactory.js';
import { initRenderer, render } from './renderGrid.js';
import { bindControls } from './bindControls.js';
import { saveEnvironment } from '../rl/storage.js';
import { createEnvironmentFromScenario, getScenarioDefinitions, DEFAULT_SCENARIO_ID } from '../rl/environmentPresets.js';
import { RLTrainer } from '../rl/training.js';

const supportsWorker = typeof Worker !== 'undefined';

const gridEl = document.getElementById('grid');
const gridSizeInput = document.getElementById('grid-size');
const stepPenaltyInput = document.getElementById('step-penalty');
const obstaclePenaltyInput = document.getElementById('obstacle-penalty');
const goalRewardInput = document.getElementById('goal-reward');
const maxStepsInput = document.getElementById('max-steps');
const agentSelectControl = document.getElementById('agent-select');
const scenarioSelect = document.getElementById('scenario-select');
const multiAgentContainer = document.getElementById('multi-agent-controls');
const agentCountSelect = document.getElementById('agent-count');
const multiAgentList = document.getElementById('multi-agent-list');

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

function parseMaxSteps(value, fallback = 50) {
  const limit = parseInt(value, 10);
  if (!Number.isFinite(limit) || limit <= 0) {
    return fallback;
  }
  return limit;
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
const initialMaxSteps = parseMaxSteps(maxStepsInput?.value, 50);
if (maxStepsInput) {
  maxStepsInput.value = initialMaxSteps;
}

let trainer;
let agent;

const AGENT_MARKER_CLASSES = [
  'agent-marker-primary',
  'agent-marker-secondary',
  'agent-marker-tertiary',
  'agent-marker-quaternary'
];

const AGENT_CHIP_CLASSES = [
  'agent-chip-primary',
  'agent-chip-secondary',
  'agent-chip-tertiary',
  'agent-chip-quaternary'
];

const multiAgentState = {
  agentCount: 1,
  entries: new Map(),
  states: new Map(),
  integrationEnabled: false,
  isRunning: false,
  trainerMethodProxies: null,
  latestBaseMetrics: null,
  latestDisplayMetrics: null,
  originalLiveChart: null
};

updateTrackedAgentState(1, env.getState());
syncMultiAgentAvailability(env.size);

function isTrainerRunningInstance(targetTrainer = trainer) {
  if (!targetTrainer) return false;
  const running = targetTrainer.isRunning;
  if (typeof running === 'function') {
    try {
      return Boolean(running.call(targetTrainer));
    } catch (err) {
      return false;
    }
  }
  if (typeof running === 'boolean') {
    return running;
  }
  return Boolean(running);
}

function getAgentColorClass(index) {
  const idx = Math.max(1, index);
  return AGENT_MARKER_CLASSES[(idx - 1) % AGENT_MARKER_CLASSES.length];
}

function getAgentChipClass(index) {
  const idx = Math.max(1, index);
  return AGENT_CHIP_CLASSES[(idx - 1) % AGENT_CHIP_CLASSES.length];
}

function getAgentLabel(index) {
  return `Agent ${index}`;
}

function toPosition(state) {
  if (!state) return null;
  if (ArrayBuffer.isView(state) || Array.isArray(state)) {
    const [sx, sy] = state;
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) return null;
    return { x: Math.trunc(sx), y: Math.trunc(sy) };
  }
  if (typeof state === 'object') {
    if (Number.isFinite(state.x) && Number.isFinite(state.y)) {
      return { x: Math.trunc(state.x), y: Math.trunc(state.y) };
    }
    if (Array.isArray(state.state) || ArrayBuffer.isView(state.state)) {
      return toPosition(state.state);
    }
  }
  return null;
}

function updateTrackedAgentState(index, state) {
  const position = toPosition(state);
  if (!position) return;
  multiAgentState.states.set(index, {
    position,
    colorClass: getAgentColorClass(index),
    label: getAgentLabel(index)
  });
}

function removeTrackedAgentState(index) {
  multiAgentState.states.delete(index);
}

function collectAgentRenderStates() {
  const entries = Array.from(multiAgentState.states.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, value]) => ({
      position: value.position,
      colorClass: value.colorClass,
      label: value.label
    }));
  return entries.map(entry => ({
    position: entry.position,
    colorClass: entry.colorClass,
    label: entry.label
  }));
}

function renderCurrentAgents() {
  const states = collectAgentRenderStates();
  if (states.length > 0) {
    render(states.map(entry => ({
      position: entry.position,
      colorClass: entry.colorClass,
      label: entry.label
    })));
    return;
  }
  if (env && typeof env.getState === 'function') {
    render(env.getState());
  }
}

function handleEnvironmentChange(updatedEnv = env) {
  if (!trainer) return;
  const nextEnv = updatedEnv || env;
  if (typeof trainer.setEnvironment === 'function') {
    trainer.setEnvironment(nextEnv);
  } else if (nextEnv) {
    trainer.env = nextEnv;
  }
  if (nextEnv && typeof nextEnv.getState === 'function') {
    updateTrackedAgentState(1, nextEnv.getState());
  }
  syncAdditionalEnvironments(nextEnv);
  renderCurrentAgents();
  updateDisplayedMetrics(multiAgentState.latestBaseMetrics || trainer?.metrics || null);
}

initRenderer(env, gridEl, env.size, handleEnvironmentChange);

const policySelect = document.getElementById('policy-select');
const lambdaSlider = document.getElementById('lambda-slider');
const learningRateSlider = document.getElementById('learning-rate-slider');

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

function computeAggregatedMetrics(baseMetrics) {
  const metricsList = [];
  if (baseMetrics) metricsList.push(baseMetrics);
  multiAgentState.entries.forEach(entry => {
    if (entry.lastMetrics) {
      metricsList.push(entry.lastMetrics);
    } else if (entry.trainer?.metrics) {
      metricsList.push(entry.trainer.metrics);
    }
  });
  if (metricsList.length === 0) {
    return baseMetrics || null;
  }
  const aggregate = {
    episode: 0,
    steps: 0,
    cumulativeReward: 0,
    epsilon: 0
  };
  let epsilonSamples = 0;
  for (const metrics of metricsList) {
    if (!metrics) continue;
    if (Number.isFinite(metrics.episode)) {
      aggregate.episode = Math.max(aggregate.episode, metrics.episode);
    }
    if (Number.isFinite(metrics.steps)) {
      aggregate.steps += metrics.steps;
    }
    if (Number.isFinite(metrics.cumulativeReward)) {
      aggregate.cumulativeReward += metrics.cumulativeReward;
    }
    if (Number.isFinite(metrics.epsilon)) {
      aggregate.epsilon += metrics.epsilon;
      epsilonSamples += 1;
    }
  }
  const count = metricsList.length || 1;
  aggregate.steps = aggregate.steps / count;
  aggregate.cumulativeReward = aggregate.cumulativeReward / count;
  aggregate.epsilon = epsilonSamples > 0 ? aggregate.epsilon / epsilonSamples : 0;
  if (!Number.isFinite(aggregate.episode) || aggregate.episode <= 0) {
    aggregate.episode = baseMetrics?.episode ?? 1;
  }
  return aggregate;
}

function updateDisplayedMetrics(baseMetrics) {
  const aggregated = multiAgentState.agentCount > 1
    ? computeAggregatedMetrics(baseMetrics)
    : baseMetrics;
  if (!aggregated) return;
  metricsEls.episode.textContent = Math.max(1, Math.round(aggregated.episode));
  metricsEls.steps.textContent = Math.max(0, Math.round(aggregated.steps ?? 0));
  metricsEls.reward.textContent = Number(aggregated.cumulativeReward ?? 0).toFixed(2);
  metricsEls.epsilon.textContent = Number(aggregated.epsilon ?? 0).toFixed(2);
  if (metricsEls.epsilonSlider) {
    metricsEls.epsilonSlider.value = Number(aggregated.epsilon ?? 0);
  }
  if (metricsEls.epsilonValue) {
    metricsEls.epsilonValue.textContent = Number(aggregated.epsilon ?? 0).toFixed(2);
  }
  multiAgentState.latestDisplayMetrics = aggregated;
}

function getBaseAgentInstance() {
  if (trainer && trainer.agent) {
    return trainer.agent;
  }
  if (agent) return agent;
  return baseAgent;
}

function applySharedSettings(targetAgent) {
  if (!targetAgent) return;
  const base = getBaseAgentInstance();
  if (!base) return;
  const epsilonValue = metricsEls.epsilonSlider ? parseFloat(metricsEls.epsilonSlider.value) : base.epsilon;
  const lambdaValue = lambdaSlider ? parseFloat(lambdaSlider.value) : base.lambda;
  const fields = [
    'epsilon',
    'epsilonDecay',
    'minEpsilon',
    'learningRate',
    'gamma',
    'lambda',
    'temperature',
    'ucbC',
    'alpha',
    'beta',
    'planningSteps',
    'exploringStarts',
    'initialValue',
    'alphaCritic',
    'alphaActor'
  ];
  for (const field of fields) {
    if (base[field] !== undefined && targetAgent[field] !== undefined) {
      targetAgent[field] = base[field];
    }
  }
  if (targetAgent.epsilon !== undefined && Number.isFinite(epsilonValue)) {
    targetAgent.epsilon = epsilonValue;
  }
  if (targetAgent.lambda !== undefined && Number.isFinite(lambdaValue)) {
    targetAgent.lambda = lambdaValue;
  }
  if (policySelect && targetAgent.policy !== undefined) {
    targetAgent.policy = policySelect.value;
  }
}

function applySharedSettingsToAdditionalAgents() {
  multiAgentState.entries.forEach(entry => {
    if (entry.agent) {
      applySharedSettings(entry.agent);
    }
  });
}

function createEnvironmentClone(baseEnvironment = env) {
  const source = baseEnvironment || env;
  if (!source) return null;
  const rewards = typeof source.getRewardConfig === 'function'
    ? source.getRewardConfig()
    : undefined;
  const scenarioConfig = cloneScenarioConfig(source);
  const options = {
    size: source.size,
    obstacles: cloneObstacles(source.obstacles)
  };
  if (rewards) {
    options.rewards = { ...rewards };
  }
  if (scenarioConfig !== undefined) {
    options.scenarioConfig = scenarioConfig;
  }
  const scenarioId = source.scenarioId ?? currentScenarioId;
  return createEnvironmentFromScenario(scenarioId, options);
}

function createAdditionalAgentEntry(index, type) {
  const envClone = createEnvironmentClone(env);
  if (!envClone) return null;
  const agentInstance = createAgent(type, {
    policy: policySelect?.value ?? 'epsilon-greedy',
    lambda: lambdaSlider ? parseFloat(lambdaSlider.value) : 0
  });
  applySharedSettings(agentInstance);
  const interval = trainer?.intervalMs ?? 100;
  const limit = trainer?.maxSteps ?? parseMaxSteps(maxStepsInput?.value, initialMaxSteps);
  const additionalTrainer = new RLTrainer(agentInstance, envClone, {
    intervalMs: interval,
    maxSteps: limit,
    liveChart: null,
    onStep: (state, reward, done, metrics) => {
      handleAdditionalProgress(index, state, reward, done, metrics);
    }
  });
  const entry = {
    index,
    type,
    agent: agentInstance,
    env: envClone,
    trainer: additionalTrainer,
    colorClass: getAgentColorClass(index),
    chipClass: getAgentChipClass(index),
    lastMetrics: additionalTrainer.metrics
  };
  multiAgentState.entries.set(index, entry);
  if (typeof additionalTrainer.resetTrainerState === 'function') {
    additionalTrainer.resetTrainerState();
  } else if (typeof additionalTrainer.reset === 'function') {
    additionalTrainer.reset();
  }
  updateTrackedAgentState(index, envClone.getState());
  return entry;
}

function removeAdditionalAgent(index) {
  const entry = multiAgentState.entries.get(index);
  if (!entry) return;
  if (entry.trainer && typeof entry.trainer.pause === 'function') {
    entry.trainer.pause();
  }
  multiAgentState.entries.delete(index);
  removeTrackedAgentState(index);
}

function resetAdditionalAgentStates() {
  multiAgentState.entries.forEach(entry => {
    if (entry.env && typeof entry.env.getState === 'function') {
      updateTrackedAgentState(entry.index, entry.env.getState());
    }
  });
}

function handleAdditionalAgentSelection(index, type) {
  removeAdditionalAgent(index);
  const entry = createAdditionalAgentEntry(index, type);
  if (!entry) return;
  entry.type = type;
  if (multiAgentState.isRunning && entry.trainer && typeof entry.trainer.start === 'function') {
    entry.trainer.start();
  }
  updateDisplayedMetrics(multiAgentState.latestBaseMetrics || trainer?.metrics || null);
  renderCurrentAgents();
}

function renderMultiAgentList() {
  if (!multiAgentList) return;
  multiAgentList.innerHTML = '';
  for (let index = 2; index <= multiAgentState.agentCount; index += 1) {
    const entry = multiAgentState.entries.get(index);
    const item = document.createElement('div');
    item.className = 'multi-agent-item';
    const chip = document.createElement('span');
    chip.className = `agent-chip ${getAgentChipClass(index)}`;
    chip.textContent = index;
    const block = document.createElement('div');
    block.className = 'control-block';
    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = getAgentLabel(index);
    const select = document.createElement('select');
    select.className = 'control-input';
    select.dataset.agentIndex = String(index);
    if (agentSelectControl) {
      select.innerHTML = agentSelectControl.innerHTML;
    }
    const selectedType = entry?.type ?? agentSelectControl?.value ?? 'rl';
    select.value = selectedType;
    select.addEventListener('change', event => {
      handleAdditionalAgentSelection(index, event.target.value);
    });
    block.append(label, select);
    item.append(chip, block);
    multiAgentList.appendChild(item);
  }
}

function ensureMultiAgentIntegration() {
  if (!trainer || multiAgentState.integrationEnabled) return;
  const originals = {};
  const currentlyRunning = isTrainerRunningInstance(trainer);
  if (typeof trainer.start === 'function') {
    originals.start = trainer.start.bind(trainer);
    trainer.start = () => {
      originals.start();
      multiAgentState.isRunning = true;
      multiAgentState.entries.forEach(entry => {
        if (entry.trainer && typeof entry.trainer.start === 'function') {
          entry.trainer.start();
        }
      });
    };
  }
  if (typeof trainer.pause === 'function') {
    originals.pause = trainer.pause.bind(trainer);
    trainer.pause = () => {
      originals.pause();
      multiAgentState.isRunning = false;
      multiAgentState.entries.forEach(entry => {
        if (entry.trainer && typeof entry.trainer.pause === 'function') {
          entry.trainer.pause();
        }
      });
    };
  }
  if (typeof trainer.reset === 'function') {
    originals.reset = trainer.reset.bind(trainer);
    trainer.reset = () => {
      originals.reset();
      multiAgentState.isRunning = false;
      multiAgentState.latestBaseMetrics = trainer.metrics || null;
      multiAgentState.entries.forEach(entry => {
        if (entry.trainer && typeof entry.trainer.reset === 'function') {
          entry.trainer.reset();
        }
      });
      resetAdditionalAgentStates();
      renderCurrentAgents();
      updateDisplayedMetrics(multiAgentState.latestBaseMetrics || trainer.metrics || null);
    };
  }
  if (typeof trainer.resetTrainerState === 'function') {
    originals.resetTrainerState = trainer.resetTrainerState.bind(trainer);
    trainer.resetTrainerState = () => {
      originals.resetTrainerState();
      multiAgentState.latestBaseMetrics = trainer.metrics || null;
      multiAgentState.entries.forEach(entry => {
        if (entry.trainer && typeof entry.trainer.resetTrainerState === 'function') {
          entry.trainer.resetTrainerState();
        } else if (entry.trainer && typeof entry.trainer.reset === 'function') {
          entry.trainer.reset();
        }
      });
      resetAdditionalAgentStates();
      renderCurrentAgents();
    };
  }
  if (typeof trainer.setIntervalMs === 'function') {
    originals.setIntervalMs = trainer.setIntervalMs.bind(trainer);
    trainer.setIntervalMs = ms => {
      originals.setIntervalMs(ms);
      multiAgentState.entries.forEach(entry => {
        if (entry.trainer && typeof entry.trainer.setIntervalMs === 'function') {
          entry.trainer.setIntervalMs(ms);
        }
      });
    };
  }
  if (typeof trainer.setMaxSteps === 'function') {
    originals.setMaxSteps = trainer.setMaxSteps.bind(trainer);
    trainer.setMaxSteps = limit => {
      originals.setMaxSteps(limit);
      multiAgentState.entries.forEach(entry => {
        if (entry.trainer && typeof entry.trainer.setMaxSteps === 'function') {
          entry.trainer.setMaxSteps(limit);
        } else if (entry.trainer) {
          entry.trainer.maxSteps = limit;
        }
      });
    };
  }
  if (typeof trainer.setEnvironment === 'function') {
    originals.setEnvironment = trainer.setEnvironment.bind(trainer);
    trainer.setEnvironment = newEnv => {
      originals.setEnvironment(newEnv);
      syncAdditionalEnvironments(newEnv);
    };
  }
  multiAgentState.trainerMethodProxies = originals;
  if (trainer.liveChart !== undefined) {
    multiAgentState.originalLiveChart = trainer.liveChart;
    trainer.liveChart = null;
  }
  multiAgentState.integrationEnabled = true;
  multiAgentState.isRunning = currentlyRunning;
  if (currentlyRunning) {
    multiAgentState.entries.forEach(entry => {
      if (entry.trainer && typeof entry.trainer.start === 'function') {
        entry.trainer.start();
      }
    });
  }
}

function disableMultiAgentIntegration() {
  if (!trainer || !multiAgentState.integrationEnabled) return;
  const originals = multiAgentState.trainerMethodProxies || {};
  if (originals.start) trainer.start = originals.start;
  if (originals.pause) trainer.pause = originals.pause;
  if (originals.reset) trainer.reset = originals.reset;
  if (originals.resetTrainerState) trainer.resetTrainerState = originals.resetTrainerState;
  if (originals.setIntervalMs) trainer.setIntervalMs = originals.setIntervalMs;
  if (originals.setMaxSteps) trainer.setMaxSteps = originals.setMaxSteps;
  if (originals.setEnvironment) trainer.setEnvironment = originals.setEnvironment;
  multiAgentState.trainerMethodProxies = null;
  multiAgentState.integrationEnabled = false;
  multiAgentState.isRunning = false;
  if (multiAgentState.originalLiveChart !== null && trainer.liveChart !== undefined) {
    trainer.liveChart = multiAgentState.originalLiveChart;
    multiAgentState.originalLiveChart = null;
  }
}

function syncAdditionalEnvironments(baseEnvironment = env) {
  multiAgentState.entries.forEach(entry => {
    const envClone = createEnvironmentClone(baseEnvironment);
    if (!envClone) return;
    entry.env = envClone;
    if (entry.trainer) {
      if (typeof entry.trainer.setEnvironment === 'function') {
        entry.trainer.setEnvironment(envClone);
      } else {
        entry.trainer.env = envClone;
      }
      if (typeof entry.trainer.reset === 'function') {
        entry.trainer.reset();
      }
      if (multiAgentState.isRunning && typeof entry.trainer.start === 'function') {
        entry.trainer.start();
      }
      entry.lastMetrics = entry.trainer.metrics;
    }
    updateTrackedAgentState(entry.index, envClone.getState());
  });
}

function syncMultiAgentAvailability(size) {
  if (!multiAgentContainer || !agentCountSelect) return;
  if (size >= 10) {
    multiAgentContainer.classList.remove('is-hidden');
  } else {
    multiAgentContainer.classList.add('is-hidden');
    if (multiAgentState.agentCount !== 1) {
      setAgentCount(1);
    } else {
      const extra = Array.from(multiAgentState.entries.keys());
      for (const index of extra) {
        removeAdditionalAgent(index);
      }
      renderMultiAgentList();
      disableMultiAgentIntegration();
    }
  }
}

function setAgentCount(count) {
  const maxAgents = AGENT_MARKER_CLASSES.length;
  const nextCount = Math.max(1, Math.min(maxAgents, Number(count) || 1));
  if (agentCountSelect) {
    agentCountSelect.value = String(nextCount);
  }
  if (multiAgentState.agentCount === nextCount) {
    renderMultiAgentList();
    return;
  }
  multiAgentState.agentCount = nextCount;
  const baseTrainerRunning = isTrainerRunningInstance(trainer);
  if (nextCount > 1) {
    ensureMultiAgentIntegration();
    multiAgentState.isRunning = baseTrainerRunning;
  } else {
    disableMultiAgentIntegration();
  }
  const existing = Array.from(multiAgentState.entries.keys());
  for (const index of existing) {
    if (index > nextCount) {
      removeAdditionalAgent(index);
    }
  }
  const shouldStartAgents = multiAgentState.isRunning;
  for (let index = 2; index <= nextCount; index += 1) {
    if (!multiAgentState.entries.has(index)) {
      const type = agentSelectControl?.value ?? 'rl';
      const entry = createAdditionalAgentEntry(index, type);
      if (entry && shouldStartAgents && entry.trainer && typeof entry.trainer.start === 'function') {
        entry.trainer.start();
      }
    }
  }
  resetAdditionalAgentStates();
  renderMultiAgentList();
  multiAgentState.latestBaseMetrics = trainer?.metrics || multiAgentState.latestBaseMetrics;
  updateDisplayedMetrics(multiAgentState.latestBaseMetrics || trainer?.metrics || null);
  renderCurrentAgents();
}

function handleAdditionalProgress(index, state, reward, done, metrics) {
  const entry = multiAgentState.entries.get(index);
  if (entry) {
    entry.lastMetrics = metrics;
  }
  updateTrackedAgentState(index, state);
  const baseMetrics = multiAgentState.latestBaseMetrics || trainer?.metrics || null;
  updateDisplayedMetrics(baseMetrics);
  renderCurrentAgents();
}

function handleProgress(state, reward, done, metrics) {
  updateTrackedAgentState(1, state);
  multiAgentState.latestBaseMetrics = metrics;
  updateDisplayedMetrics(metrics);
  if (multiAgentState.agentCount > 1 && liveChart) {
    const aggregated = computeAggregatedMetrics(metrics);
    if (aggregated) {
      liveChart.push(aggregated.cumulativeReward ?? 0, aggregated.epsilon ?? 0);
    }
  }
  renderCurrentAgents();
}

if (supportsWorker) {
  trainer = createWorkerTrainer(baseAgent, env, {
    intervalMs: 100,
    maxSteps: initialMaxSteps,
    liveChart,
    onProgress: handleProgress
  });
  agent = trainer.agent;
} else {
  trainer = new RLTrainer(baseAgent, env, {
    intervalMs: 100,
    maxSteps: initialMaxSteps,
    liveChart,
    onStep: handleProgress
  });
  agent = baseAgent;
}

if (multiAgentState.agentCount > 1) {
  ensureMultiAgentIntegration();
}
renderMultiAgentList();
multiAgentState.latestBaseMetrics = trainer?.metrics || null;

if (agentCountSelect) {
  agentCountSelect.addEventListener('change', e => {
    setAgentCount(parseInt(e.target.value, 10));
  });
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
  updateTrackedAgentState(1, env.getState());
  syncMultiAgentAvailability(env.size);
  initRenderer(env, gridEl, env.size, handleEnvironmentChange);
  if (trainer) {
    if (typeof trainer.setEnvironment === 'function') {
      trainer.setEnvironment(env);
    } else {
      trainer.env = env;
    }
    trainer.reset();
    multiAgentState.latestBaseMetrics = trainer.metrics || null;
  }
  syncAdditionalEnvironments(env);
  resetAdditionalAgentStates();
  saveEnvironment(env);
  updateDisplayedMetrics(trainer?.metrics || multiAgentState.latestDisplayMetrics || null);
  renderCurrentAgents();
  return env;
}

gridSizeInput.addEventListener('change', e => {
  const newSize = parseGridSize(e.target.value);
  rebuildEnvironment({ size: newSize });
});

const renderForControls = state => {
  if (state) {
    updateTrackedAgentState(1, state);
  }
  renderCurrentAgents();
};

bindControls(trainer, agent, renderForControls, () => env, rebuildEnvironment);

if (policySelect) {
  policySelect.addEventListener('change', () => {
    applySharedSettingsToAdditionalAgents();
  });
}

if (agentSelectControl) {
  agentSelectControl.addEventListener('change', () => {
    applySharedSettingsToAdditionalAgents();
  });
}

if (metricsEls.epsilonSlider) {
  metricsEls.epsilonSlider.addEventListener('input', e => {
    const value = parseFloat(e.target.value);
    if (!Number.isFinite(value)) return;
    multiAgentState.entries.forEach(entry => {
      if (entry.agent && entry.agent.epsilon !== undefined) {
        entry.agent.epsilon = value;
      }
    });
  });
}

if (learningRateSlider) {
  learningRateSlider.addEventListener('input', e => {
    const value = parseFloat(e.target.value);
    if (!Number.isFinite(value)) return;
    multiAgentState.entries.forEach(entry => {
      if (entry.agent && entry.agent.learningRate !== undefined) {
        entry.agent.learningRate = value;
      }
    });
  });
}

if (lambdaSlider) {
  lambdaSlider.addEventListener('input', e => {
    const value = parseFloat(e.target.value);
    if (!Number.isFinite(value)) return;
    multiAgentState.entries.forEach(entry => {
      if (entry.agent && entry.agent.lambda !== undefined) {
        entry.agent.lambda = value;
      }
    });
  });
}

updateDisplayedMetrics(trainer?.metrics || multiAgentState.latestDisplayMetrics || null);

saveEnvironment(env);
renderCurrentAgents();

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
  const defaultMaxSteps = Number.isFinite(options.maxSteps) && options.maxSteps > 0
    ? Math.trunc(options.maxSteps)
    : 50;

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
    maxSteps: defaultMaxSteps,
    metrics,
    state: typeof currentEnv.getState === 'function' ? currentEnv.getState() : null,
    episodeRewards: [],
    agent: null,
    isRunning: false,
    start() {
      this.isRunning = true;
      worker.postMessage({ type: 'start' });
    },
    pause() {
      this.isRunning = false;
      worker.postMessage({ type: 'pause' });
    },
    reset() {
      this.isRunning = false;
      worker.postMessage({ type: 'reset' });
    },
    resetTrainerState() {
      this.isRunning = false;
      worker.postMessage({ type: 'resetTrainerState' });
    },
    setIntervalMs(ms) {
      this.intervalMs = ms;
      worker.postMessage({ type: 'interval', payload: ms });
    },
    setMaxSteps(limit) {
      const parsed = Number(limit);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        return;
      }
      const nextLimit = Math.trunc(parsed);
      this.maxSteps = nextLimit;
      worker.postMessage({ type: 'maxSteps', payload: nextLimit });
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
    const scenarioId = currentEnv?.scenarioId ?? DEFAULT_SCENARIO_ID;
    const scenarioConfig = cloneScenarioConfig(currentEnv);
    worker.postMessage({
      type: 'config',
      payload: {
        agent: {
          type: rawAgent.__factoryType || 'rl',
          params: serializeAgent(rawAgent),
          revision: agentRevision
        },
        env: {
          scenarioId,
          size: currentEnv.size,
          obstacles: cloneObstacles(currentEnv.obstacles),
          ...(rewards ? { rewards } : {}),
          ...(scenarioConfig !== undefined ? { scenarioConfig } : {})
        },
        trainer: {
          intervalMs: trainerProxy.intervalMs,
          maxSteps: trainerProxy.maxSteps
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
    if (options.liveChart && payload.metrics && multiAgentState.agentCount <= 1) {
      options.liveChart.push(payload.metrics.cumulativeReward, payload.metrics.epsilon);
    }
  };

  sendConfig();

  return trainerProxy;
}
