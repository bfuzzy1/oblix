import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';
import { LiveChart } from '../src/ui/liveChart.js';

class MockContext {
  constructor() {
    this.clearCount = 0;
  }
  clearRect() { this.clearCount++; }
  beginPath() {}
  moveTo() {}
  lineTo() {}
  stroke() {}
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
  const chart = new LiveChart(canvas);
  const trainer = new RLTrainer(agent, env, { liveChart: chart });
  trainer.state = env.reset();
  await trainer.step();
  await trainer.step();
  assert.deepStrictEqual(chart.rewards.map(v => +v.toFixed(2)), [-0.01, 0.99, 0]);
  assert.deepStrictEqual(chart.epsilons.map(v => +v.toFixed(2)), [0.1, 0.1, 0.1]);
  assert.strictEqual(canvas.ctx.clearCount, 3);
}
