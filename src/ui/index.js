import { GridWorldEnvironment } from '../rl/environment.js';
import { RLTrainer } from '../rl/training.js';
import { LiveChart } from './liveChart.js';
import { createAgent } from './agentFactory.js';
import { initRenderer, render } from './renderGrid.js';
import { bindControls } from './bindControls.js';

const size = 5;
const env = new GridWorldEnvironment(size);
const gridEl = document.getElementById('grid');
initRenderer(env, gridEl, size);

const policySelect = document.getElementById('policy-select');
const lambdaSlider = document.getElementById('lambda-slider');

let agent = createAgent('rl', {
  policy: policySelect.value,
  lambda: parseFloat(lambdaSlider.value)
});

const liveChart = new LiveChart(document.getElementById('liveChart'));
const trainer = new RLTrainer(agent, env, {
  intervalMs: 100,
  liveChart,
  onStep: (state, reward, done, metrics) => {
    render(state);
    document.getElementById('episode').textContent = metrics.episode;
    document.getElementById('steps').textContent = metrics.steps;
    document.getElementById('reward').textContent = metrics.cumulativeReward.toFixed(2);
    document.getElementById('epsilon').textContent = metrics.epsilon.toFixed(2);

    document.getElementById('epsilon-slider').value = metrics.epsilon;
    document.getElementById('epsilon-value').textContent = metrics.epsilon.toFixed(2);
  }
});

bindControls(trainer, agent, render);

render(env.reset());
