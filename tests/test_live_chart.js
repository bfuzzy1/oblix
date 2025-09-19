import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';
import { LiveChart } from '../src/ui/liveChart.js';

class MockContext {
  constructor() {
    this.clearCount = 0;
    this.texts = [];
    this.strokeStyles = [];
    this.fillStyle = '#000';
    this.strokeStyle = '#000';
  }
  clearRect() { this.clearCount++; }
  beginPath() {}
  moveTo() {}
  lineTo() {}
  stroke() { this.strokeStyles.push(this.strokeStyle); }
  fillRect() {}
  fillText(text) { this.texts.push({ text, style: this.fillStyle }); }
}

class MockCanvas {
  constructor() {
    this.width = 100;
    this.height = 50;
    this.ctx = new MockContext();
  }
  getContext() { return this.ctx; }
}

export async function run(assert) {
  const env = new GridWorldEnvironment(2);
  class StubAgent {
    constructor() {
      this.epsilon = 0.1;
      this.actions = [3, 1];
      this.i = 0;
    }
    act() { return this.actions[this.i++]; }
    learn() {}
  }
  const agent = new StubAgent();
  const canvas = new MockCanvas();
  const chart = new LiveChart(canvas, 2);
  const trainer = new RLTrainer(agent, env, { liveChart: chart });
  await trainer.step();
  await trainer.step();
  assert.deepStrictEqual(chart.rewards.map(v => +v.toFixed(2)), [0, -0.01, 0.99, 0]);
  assert.deepStrictEqual(chart.avgRewards.map(v => +v.toFixed(2)), [0, -0.01, 0.49, 0.49]);
  assert.deepStrictEqual(chart.epsilons.map(v => +v.toFixed(2)), [0.1, 0.1, 0.1, 0.1]);
  assert.strictEqual(canvas.ctx.clearCount, 4);
  const expectedStrokes = ['rgba(148, 163, 184, 0.25)', '#5eead4', '#fbbf24', '#38bdf8'];
  for (let i = 0; i < canvas.ctx.strokeStyles.length; i += 4) {
    assert.deepStrictEqual(
      canvas.ctx.strokeStyles.slice(i, i + 4),
      expectedStrokes
    );
  }
  const rewardEntry = canvas.ctx.texts.find(t => t.text === 'Reward');
  const avgEntry = canvas.ctx.texts.find(t => t.text === 'Avg Reward');
  const epsilonEntry = canvas.ctx.texts.find(t => t.text === 'Epsilon');
  assert.ok(rewardEntry && avgEntry && epsilonEntry);
  assert.strictEqual(rewardEntry.style, '#e2e8f0');
  assert.strictEqual(avgEntry.style, '#e2e8f0');
  assert.strictEqual(epsilonEntry.style, '#e2e8f0');
}
