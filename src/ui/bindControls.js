import { createAgent } from './agentFactory.js';
import { saveAgent, loadAgent, saveEnvironment, loadEnvironment } from '../rl/storage.js';
import { DEFAULT_REWARD_CONFIG } from '../rl/environment.js';

function getElements() {
  return {
    agentSelect: document.getElementById('agent-select'),
    policySelect: document.getElementById('policy-select'),
    epsilonSlider: document.getElementById('epsilon-slider'),
    epsilonValue: document.getElementById('epsilon-value'),
    intervalSlider: document.getElementById('interval-slider'),
    intervalValue: document.getElementById('interval-value'),
    learningRateSlider: document.getElementById('learning-rate-slider'),
    learningRateValue: document.getElementById('learning-rate-value'),
    lambdaSlider: document.getElementById('lambda-slider'),
    lambdaValue: document.getElementById('lambda-value'),
    stepPenaltyInput: document.getElementById('step-penalty'),
    obstaclePenaltyInput: document.getElementById('obstacle-penalty'),
    goalRewardInput: document.getElementById('goal-reward')
  };
}

function hasLearningRate(agent) {
  return agent && agent.learningRate !== undefined;
}

function syncLearningRate(agent, els) {
  if (!hasLearningRate(agent)) {
    return;
  }
  els.learningRateSlider.value = agent.learningRate;
  els.learningRateValue.textContent = agent.learningRate.toFixed(2);
}

function updateLearningRateControl(agent, els) {
  if (hasLearningRate(agent)) {
    els.learningRateSlider.disabled = false;
    syncLearningRate(agent, els);
  } else {
    els.learningRateSlider.disabled = true;
    els.learningRateValue.textContent = 'N/A';
  }
}

function syncLambda(agent, els) {
  const val = agent.lambda ?? parseFloat(els.lambdaSlider.value);
  els.lambdaSlider.value = val;
  els.lambdaValue.textContent = val.toFixed(2);
}

function getRewardConfig(env) {
  if (!env) return { ...DEFAULT_REWARD_CONFIG };
  if (typeof env.getRewardConfig === 'function') {
    return env.getRewardConfig();
  }
  return {
    stepPenalty: env.stepPenalty ?? DEFAULT_REWARD_CONFIG.stepPenalty,
    obstaclePenalty: env.obstaclePenalty ?? DEFAULT_REWARD_CONFIG.obstaclePenalty,
    goalReward: env.goalReward ?? DEFAULT_REWARD_CONFIG.goalReward
  };
}

function syncEnvironmentControls(env, els) {
  if (!els.stepPenaltyInput || !els.obstaclePenaltyInput || !els.goalRewardInput) {
    return;
  }
  const rewards = getRewardConfig(env);
  els.stepPenaltyInput.value = Number(rewards.stepPenalty);
  els.obstaclePenaltyInput.value = Number(rewards.obstaclePenalty);
  els.goalRewardInput.value = Number(rewards.goalReward);
}

function initializeControls(trainer, agent, env, els) {
  els.epsilonSlider.value = agent.epsilon;
  els.epsilonValue.textContent = agent.epsilon.toFixed(2);
  els.policySelect.value = agent.policy;
  els.intervalSlider.value = trainer.intervalMs;
  els.intervalValue.textContent = trainer.intervalMs;
  updateLearningRateControl(agent, els);
  syncLambda(agent, els);
  syncEnvironmentControls(env, els);
}

function bindAgentSelection(trainer, els, getAgent, setAgent, getEnv) {
  els.agentSelect.addEventListener('change', e => {
    const newAgent = createAgent(e.target.value, {
      policy: els.policySelect.value,
      lambda: parseFloat(els.lambdaSlider.value)
    });
    const assigned = setAgent(newAgent);
    trainer.reset();
    initializeControls(trainer, assigned, getEnv(), els);
  });
}

function bindPolicySelection(els, getAgent) {
  els.policySelect.addEventListener('change', e => {
    getAgent().policy = e.target.value;
  });
}

function bindSliders(trainer, els, getAgent) {
  els.epsilonSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    const agent = getAgent();
    agent.epsilon = val;
    trainer.metrics.epsilon = val;
    els.epsilonValue.textContent = val.toFixed(2);
    document.getElementById('epsilon').textContent = val.toFixed(2);
  });

  els.intervalSlider.addEventListener('input', e => {
    const val = parseInt(e.target.value, 10);
    trainer.setIntervalMs(val);
    els.intervalValue.textContent = val;
  });

  els.learningRateSlider.addEventListener('input', e => {
    const agent = getAgent();
    if (agent.learningRate === undefined) {
      return;
    }
    const val = parseFloat(e.target.value);
    agent.learningRate = val;
    els.learningRateValue.textContent = val.toFixed(2);
  });

  els.lambdaSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    const agent = getAgent();
    agent.lambda = val;
    els.lambdaValue.textContent = val.toFixed(2);
  });
}

function parseRewardInput(value, fallback) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function readRewardInputs(els, env) {
  const current = getRewardConfig(env);
  return {
    stepPenalty: parseRewardInput(els.stepPenaltyInput?.value, current.stepPenalty),
    obstaclePenalty: parseRewardInput(els.obstaclePenaltyInput?.value, current.obstaclePenalty),
    goalReward: parseRewardInput(els.goalRewardInput?.value, current.goalReward)
  };
}

function cloneObstacles(obstacles = []) {
  return obstacles.map(o => ({ x: o.x, y: o.y }));
}

function bindEnvironmentControls(trainer, els, getAgent, getEnv, setEnv) {
  if (!els.stepPenaltyInput || !els.obstaclePenaltyInput || !els.goalRewardInput) {
    return;
  }
  const handleChange = () => {
    const env = getEnv();
    const rewards = readRewardInputs(els, env);
    const size = env?.size ?? 5;
    const obstacles = env ? cloneObstacles(env.obstacles) : [];
    setEnv(size, obstacles, rewards);
    initializeControls(trainer, getAgent(), getEnv(), els);
  };
  els.stepPenaltyInput.addEventListener('change', handleChange);
  els.obstaclePenaltyInput.addEventListener('change', handleChange);
  els.goalRewardInput.addEventListener('change', handleChange);
}

function bindPersistence(trainer, els, getAgent, setAgent, render, getEnv, setEnv) {
  document.getElementById('start').onclick = () => trainer.start();
  document.getElementById('pause').onclick = () => trainer.pause();
  document.getElementById('reset').onclick = () => {
    trainer.reset();
    initializeControls(trainer, getAgent(), getEnv(), els);
  };
  document.getElementById('save').onclick = async () => {
    let agentForSave = getAgent();
    if (typeof trainer.getAgentState === 'function') {
      const snapshot = await trainer.getAgentState();
      if (snapshot) {
        agentForSave = { toJSON: () => snapshot };
      }
    }
    saveAgent(agentForSave);
    saveEnvironment(getEnv());
  };
  document.getElementById('load').onclick = () => {
    const loaded = loadAgent(trainer);
    const assigned = setAgent(loaded, { skipTrainer: true });
    initializeControls(trainer, assigned, getEnv(), els);
    const envData = loadEnvironment();
    if (envData) {
      setEnv(envData.size, envData.obstacles, envData.rewards);
      initializeControls(trainer, getAgent(), getEnv(), els);
    }
    render(trainer.state);
  };
}

export function bindControls(trainer, agent, render, getEnv, setEnv) {
  let currentAgent = agent;
  const getAgent = () => currentAgent;
  const setAgent = (a, options = {}) => {
    let nextAgent = a;
    if (!options.skipTrainer) {
      if (typeof trainer.setAgent === 'function') {
        const result = trainer.setAgent(a);
        nextAgent = result || trainer.agent;
      } else {
        trainer.agent = a;
      }
    } else if (typeof trainer.setAgent === 'function') {
      nextAgent = trainer.agent;
    }
    currentAgent = nextAgent;
    return currentAgent;
  };
  const els = getElements();

  initializeControls(trainer, currentAgent, getEnv(), els);
  bindAgentSelection(trainer, els, getAgent, setAgent, getEnv);
  bindPolicySelection(els, getAgent);
  bindSliders(trainer, els, getAgent);
  bindEnvironmentControls(trainer, els, getAgent, getEnv, setEnv);
  bindPersistence(trainer, els, getAgent, setAgent, render, getEnv, setEnv);
}
