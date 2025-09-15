import { createAgent } from './agentFactory.js';
import { saveAgent, loadAgent, saveEnvironment, loadEnvironment } from '../rl/storage.js';

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
    lambdaValue: document.getElementById('lambda-value')
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

function initializeControls(trainer, agent, els) {
  els.epsilonSlider.value = agent.epsilon;
  els.epsilonValue.textContent = agent.epsilon.toFixed(2);
  els.policySelect.value = agent.policy;
  els.intervalSlider.value = trainer.intervalMs;
  els.intervalValue.textContent = trainer.intervalMs;
  updateLearningRateControl(agent, els);
  syncLambda(agent, els);
}

function bindAgentSelection(trainer, els, getAgent, setAgent) {
  els.agentSelect.addEventListener('change', e => {
    const newAgent = createAgent(e.target.value, {
      policy: els.policySelect.value,
      lambda: parseFloat(els.lambdaSlider.value)
    });
    const assigned = setAgent(newAgent);
    trainer.reset();
    initializeControls(trainer, assigned, els);
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

function bindPersistence(trainer, els, getAgent, setAgent, render, getEnv, setEnv) {
  document.getElementById('start').onclick = () => trainer.start();
  document.getElementById('pause').onclick = () => trainer.pause();
  document.getElementById('reset').onclick = () => {
    trainer.reset();
    initializeControls(trainer, getAgent(), els);
  };
  document.getElementById('save').onclick = () => {
    saveAgent(getAgent());
    saveEnvironment(getEnv());
  };
  document.getElementById('load').onclick = () => {
    const loaded = loadAgent(trainer);
    const assigned = setAgent(loaded, { skipTrainer: true });
    initializeControls(trainer, assigned, els);
    const envData = loadEnvironment();
    if (envData) {
      setEnv(envData.size, envData.obstacles);
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

  initializeControls(trainer, currentAgent, els);
  bindAgentSelection(trainer, els, getAgent, setAgent);
  bindPolicySelection(els, getAgent);
  bindSliders(trainer, els, getAgent);
  bindPersistence(trainer, els, getAgent, setAgent, render, getEnv, setEnv);
}
