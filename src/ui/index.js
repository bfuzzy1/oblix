import { GridWorldEnvironment } from '../rl/environment.js';
import { RLAgent } from '../rl/agent.js';
import { SarsaAgent } from '../rl/sarsaAgent.js';
import { ExpectedSarsaAgent } from '../rl/expectedSarsaAgent.js';
import { DynaQAgent } from '../rl/dynaQAgent.js';
import { RLTrainer } from '../rl/training.js';
import { saveAgent, loadAgent } from '../rl/storage.js';
import { LiveChart } from './liveChart.js';

const size = 5;
const env = new GridWorldEnvironment(size);

function createAgent(type) {
  const options = { epsilon: 1, epsilonDecay: 0.995, minEpsilon: 0.05 };
  if (type === 'sarsa') return new SarsaAgent(options);
  if (type === 'expected') return new ExpectedSarsaAgent(options);
  if (type === 'dyna') return new DynaQAgent(options);
  return new RLAgent(options);
}

let agent = createAgent('rl');
const gridEl = document.getElementById('grid');
gridEl.style.setProperty('--size', size);

const agentSelect = document.getElementById('agent-select');
const liveChart = new LiveChart(document.getElementById('liveChart'));
const epsilonSlider = document.getElementById('epsilon-slider');
const epsilonValue = document.getElementById('epsilon-value');
const intervalSlider = document.getElementById('interval-slider');
const intervalValue = document.getElementById('interval-value');
const learningRateSlider = document.getElementById('learning-rate-slider');
const learningRateValue = document.getElementById('learning-rate-value');

function syncLearningRate() {
  learningRateSlider.value = agent.learningRate;
  learningRateValue.textContent = agent.learningRate.toFixed(2);
}

function render(state) {
  gridEl.innerHTML = '';
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      if (env.isObstacle(x, y)) cell.classList.add('obstacle');
      if (x === state[0] && y === state[1]) cell.classList.add('agent');
      if (x === size - 1 && y === size - 1) cell.classList.add('goal');
      cell.addEventListener('click', () => {
        if (x === env.agentPos.x && y === env.agentPos.y) return;
        if (x === size - 1 && y === size - 1) return;
        env.toggleObstacle(x, y);
        render(env.getState());
      });
      gridEl.appendChild(cell);
    }
  }
}

const trainer = new RLTrainer(agent, env, {
  intervalMs: 100,
  liveChart,
  onStep: (state, reward, done, metrics) => {
    render(state);
    document.getElementById('episode').textContent = metrics.episode;
    document.getElementById('steps').textContent = metrics.steps;
    document.getElementById('reward').textContent = metrics.cumulativeReward.toFixed(2);
    document.getElementById('epsilon').textContent = metrics.epsilon.toFixed(2);

    epsilonSlider.value = metrics.epsilon;
    epsilonValue.textContent = metrics.epsilon.toFixed(2);
  }
});

epsilonSlider.value = agent.epsilon;
epsilonValue.textContent = agent.epsilon.toFixed(2);
intervalSlider.value = trainer.intervalMs;
intervalValue.textContent = trainer.intervalMs;
syncLearningRate();

agentSelect.addEventListener('change', e => {
  agent = createAgent(e.target.value);
  trainer.agent = agent;
  trainer.reset();
  epsilonSlider.value = agent.epsilon;
  epsilonValue.textContent = agent.epsilon.toFixed(2);
  syncLearningRate();
});

epsilonSlider.addEventListener('input', e => {
  const val = parseFloat(e.target.value);
  agent.epsilon = val;
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
  agent.learningRate = val;
  learningRateValue.textContent = val.toFixed(2);
});

document.getElementById('start').onclick = () => trainer.start();
document.getElementById('pause').onclick = () => trainer.pause();
document.getElementById('reset').onclick = () => {
  trainer.reset();
  epsilonSlider.value = agent.epsilon;
  epsilonValue.textContent = agent.epsilon.toFixed(2);
  syncLearningRate();
};
document.getElementById('save').onclick = () => saveAgent(agent);
document.getElementById('load').onclick = () => {
  agent = loadAgent(trainer);
  epsilonSlider.value = agent.epsilon;
  epsilonValue.textContent = agent.epsilon.toFixed(2);
  syncLearningRate();
  render(trainer.state);
};

render(env.reset());
