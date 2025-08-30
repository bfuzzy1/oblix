export class MetricsTracker {
  constructor(agent) {
    this.reset(agent);
  }

  reset(agent) {
    this.metrics = {
      episode: 1,
      steps: 0,
      cumulativeReward: 0,
      epsilon: agent.epsilon
    };
    this.episodeRewards = [];
  }

  update(reward, agent) {
    this.metrics.steps += 1;
    this.metrics.cumulativeReward += reward;
    this.metrics.epsilon = agent.epsilon;
    return { ...this.metrics };
  }

  endEpisode(agent) {
    this.episodeRewards.push(this.metrics.cumulativeReward);
    this.metrics.episode += 1;
    this.metrics.steps = 0;
    this.metrics.cumulativeReward = 0;
    this.metrics.epsilon = agent.epsilon;
    return { ...this.metrics };
  }

  get data() {
    return this.metrics;
  }
}
