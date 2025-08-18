export class RLTrainer {
  constructor(agent, env, options = {}) {
    this.agent = agent;
    this.env = env;
    this.maxSteps = options.maxSteps ?? 50;
    this.onStep = options.onStep || null;
    this.intervalMs = options.intervalMs ?? 100;
    this.isRunning = false;
    this.interval = null;
    this.state = null;
    this.metrics = {
      episode: 1,
      steps: 0,
      cumulativeReward: 0,
      epsilon: this.agent.epsilon
    };
  }

  async step() {
    const action = this.agent.act(this.state);
    const { state: nextState, reward, done } = this.env.step(action);
    await this.agent.learn(this.state, action, reward, nextState, done);
    this.state = nextState;
    this.metrics.steps += 1;
    this.metrics.cumulativeReward += reward;
    this.metrics.epsilon = this.agent.epsilon;
    if (this.onStep) this.onStep(this.state, reward, done, { ...this.metrics });
    if (done) {
      this.metrics.episode += 1;
      this.metrics.steps = 0;
      this.metrics.cumulativeReward = 0;
      this.metrics.epsilon = this.agent.epsilon;
      this.state = this.env.reset();
      if (this.onStep) {
        this.onStep(this.state, 0, false, { ...this.metrics });
      }
    }
  }

  start() {
    if (this.isRunning) return;
    this.state = this.env.reset();
    this.isRunning = true;
    this.interval = setInterval(async () => {
      await this.step();
    }, this.intervalMs);
  }

  setIntervalMs(ms) {
    this.intervalMs = ms;
    if (this.isRunning) {
      clearInterval(this.interval);
      this.interval = setInterval(async () => {
        await this.step();
      }, this.intervalMs);
    }
  }

  pause() {
    if (!this.isRunning) return;
    clearInterval(this.interval);
    this.isRunning = false;
  }

  reset() {
    this.pause();
    this.state = this.env.reset();
    this.metrics = {
      episode: 1,
      steps: 0,
      cumulativeReward: 0,
      epsilon: this.agent.epsilon
    };
    if (this.onStep) {
      this.onStep(this.state, 0, false, { ...this.metrics });
    }
  }

  static async trainEpisodes(agent, env, episodes = 10, maxSteps = 50) {
    for (let ep = 0; ep < episodes; ep++) {
      let state = env.reset();
      for (let st = 0; st < maxSteps; st++) {
        const action = agent.act(state);
        const { state: nextState, reward, done } = env.step(action);
        await agent.learn(state, action, reward, nextState, done);
        state = nextState;
        if (done) break;
      }
    }
  }
}
