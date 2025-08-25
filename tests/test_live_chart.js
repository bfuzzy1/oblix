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
  trainer.state = env.reset();
  await trainer.step();
  await trainer.step();
  assert.deepStrictEqual(chart.rewards.map(v => +v.toFixed(2)), [-0.01, 0.99, 0]);
  assert.deepStrictEqual(chart.avgRewards.map(v => +v.toFixed(2)), [-0.01, 0.49, 0.49]);
  assert.deepStrictEqual(chart.epsilons.map(v => +v.toFixed(2)), [0.1, 0.1, 0.1]);
  assert.strictEqual(canvas.ctx.clearCount, 4);
  const expectedStrokes = ['#ccc', '#999', '#4caf50', '#ff9800', '#2196f3'];
  for (let i = 0; i < canvas.ctx.strokeStyles.length; i += 5) {
    assert.deepStrictEqual(
      canvas.ctx.strokeStyles.slice(i, i + 5),
      expectedStrokes
    );
  }
  const rewardEntry = canvas.ctx.texts.find(t => t.text === 'Reward');
  const avgEntry = canvas.ctx.texts.find(t => t.text === 'Avg Reward');
  const epsilonEntry = canvas.ctx.texts.find(t => t.text === 'Epsilon');
  assert.ok(rewardEntry && avgEntry && epsilonEntry);
  assert.strictEqual(rewardEntry.style, '#ccc');
  assert.strictEqual(avgEntry.style, '#ccc');
  assert.strictEqual(epsilonEntry.style, '#ccc');

  const tickTexts = canvas.ctx.texts.filter(t =>
    /^[-\d]/.test(t.text) &&
    !['Reward', 'Avg Reward', 'Epsilon'].includes(t.text)
  );
  assert.ok(tickTexts.length > 0);
}
