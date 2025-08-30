import { createAgent } from './agentFactory.js';
import { saveAgent, loadAgent } from '../rl/storage.js';

export function bindControls(trainer, agent, render) {
  let currentAgent = agent;

  const agentSelect = document.getElementById('agent-select');
  const policySelect = document.getElementById('policy-select');
  const epsilonSlider = document.getElementById('epsilon-slider');
  const epsilonValue = document.getElementById('epsilon-value');
  const intervalSlider = document.getElementById('interval-slider');
  const intervalValue = document.getElementById('interval-value');
  const learningRateSlider = document.getElementById('learning-rate-slider');
  const learningRateValue = document.getElementById('learning-rate-value');
  const lambdaSlider = document.getElementById('lambda-slider');
  const lambdaValue = document.getElementById('lambda-value');

  function syncLearningRate() {
    learningRateSlider.value = currentAgent.learningRate;
    learningRateValue.textContent = currentAgent.learningRate.toFixed(2);
  }

  function syncLambda() {
    lambdaSlider.value = currentAgent.lambda ?? parseFloat(lambdaSlider.value);
    const val = parseFloat(lambdaSlider.value);
    lambdaValue.textContent = val.toFixed(2);
  }

  epsilonSlider.value = currentAgent.epsilon;
  epsilonValue.textContent = currentAgent.epsilon.toFixed(2);
  policySelect.value = currentAgent.policy;
  intervalSlider.value = trainer.intervalMs;
  intervalValue.textContent = trainer.intervalMs;
  syncLearningRate();
  syncLambda();

  agentSelect.addEventListener('change', e => {
    currentAgent = createAgent(e.target.value, {
      policy: policySelect.value,
      lambda: parseFloat(lambdaSlider.value)
    });
    trainer.agent = currentAgent;
    trainer.reset();
    epsilonSlider.value = currentAgent.epsilon;
    epsilonValue.textContent = currentAgent.epsilon.toFixed(2);
    policySelect.value = currentAgent.policy;
    syncLearningRate();
    syncLambda();
  });

  policySelect.addEventListener('change', e => {
    currentAgent.policy = e.target.value;
  });

  epsilonSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    currentAgent.epsilon = val;
    trainer.metrics.epsilon = val;
    epsilonValue.textContent = val.toFixed(2);
    document.getElementById('epsilon').textContent = val.toFixed(2);
  });

  intervalSlider.addEventListener('input', e => {
    const val = parseInt(e.target.value, 10);
    trainer.setIntervalMs(val);
    intervalValue.textContent = val;
  });

  learningRateSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    currentAgent.learningRate = val;
    learningRateValue.textContent = val.toFixed(2);
  });

  lambdaSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    currentAgent.lambda = val;
    lambdaValue.textContent = val.toFixed(2);
  });

  document.getElementById('start').onclick = () => trainer.start();
  document.getElementById('pause').onclick = () => trainer.pause();
  document.getElementById('reset').onclick = () => {
    trainer.reset();
    epsilonSlider.value = currentAgent.epsilon;
    epsilonValue.textContent = currentAgent.epsilon.toFixed(2);
    syncLearningRate();
  };
  document.getElementById('save').onclick = () => saveAgent(currentAgent);
  document.getElementById('load').onclick = () => {
    currentAgent = loadAgent(trainer);
    epsilonSlider.value = currentAgent.epsilon;
    epsilonValue.textContent = currentAgent.epsilon.toFixed(2);
    policySelect.value = currentAgent.policy;
    syncLearningRate();
    render(trainer.state);
  };
}
