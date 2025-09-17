export const DEFAULT_REWARD_CONFIG = Object.freeze({
  stepPenalty: -0.01,
  obstaclePenalty: -0.1,
  goalReward: 1
});

function normalizeNumber(value, fallback) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

export class GridWorldEnvironment {
  constructor(size = 5, obstacles = [], rewardConfig = {}) {
    this.size = size;
    this.obstacles = [];
    this.obstacleSet = new Set();
    this.scenarioId = 'classic';
    this.setRewardConfig(rewardConfig);
    this.setObstacles(obstacles);
    this.reset();
  }

  /** Reset the environment to the starting state. */
  reset() {
    this.agentPos = { x: 0, y: 0 };
    return this.getState();
  }

  /** Convert the current agent position to state Float32Array. */
  getState() {
    return new Float32Array([this.agentPos.x, this.agentPos.y]);
  }

  getGoalPosition() {
    return { x: this.size - 1, y: this.size - 1 };
  }

  isObstacle(x, y) {
    return this.obstacleSet.has(`${x},${y}`);
  }

  toggleObstacle(x, y) {
    const key = `${x},${y}`;
    if (this.obstacleSet.has(key)) {
      this.obstacleSet.delete(key);
      this.obstacles = this.obstacles.filter(o => o.x !== x || o.y !== y);
    } else {
      this.obstacles.push({ x, y });
      this.obstacleSet.add(key);
    }
  }

  setObstacles(obstacles = []) {
    this.obstacles = obstacles.map(o => ({ x: o.x, y: o.y }));
    this.obstacleSet = new Set(
      this.obstacles.map(o => `${o.x},${o.y}`)
    );
  }

  setRewardConfig(rewardConfig = {}) {
    const source = rewardConfig && typeof rewardConfig === 'object'
      ? rewardConfig
      : {};
    const merged = { ...DEFAULT_REWARD_CONFIG, ...source };
    this.stepPenalty = normalizeNumber(merged.stepPenalty, DEFAULT_REWARD_CONFIG.stepPenalty);
    this.obstaclePenalty = normalizeNumber(merged.obstaclePenalty, DEFAULT_REWARD_CONFIG.obstaclePenalty);
    this.goalReward = normalizeNumber(merged.goalReward, DEFAULT_REWARD_CONFIG.goalReward);
  }

  getRewardConfig() {
    return {
      stepPenalty: this.stepPenalty,
      obstaclePenalty: this.obstaclePenalty,
      goalReward: this.goalReward
    };
  }

  calculateReward(newX, newY, done) {
    return done ? this.goalReward : this.stepPenalty;
  }

  describeCell() {
    return { classes: [] };
  }

  /**
   * Step the environment with an action.
   * @param action 0:up,1:down,2:left,3:right
   */
  step(action) {
    let newX = this.agentPos.x;
    let newY = this.agentPos.y;
    switch (action) {
      case 0:
        if (newY > 0) newY -= 1;
        break;
      case 1:
        if (newY < this.size - 1) newY += 1;
        break;
      case 2:
        if (newX > 0) newX -= 1;
        break;
      case 3:
        if (newX < this.size - 1) newX += 1;
        break;
    }
    if (this.isObstacle(newX, newY)) {
      return { state: this.getState(), reward: this.obstaclePenalty, done: false };
    }
    this.agentPos = { x: newX, y: newY };
    const goal = this.getGoalPosition();
    const done = this.agentPos.x === goal.x && this.agentPos.y === goal.y;
    const reward = this.calculateReward(this.agentPos.x, this.agentPos.y, done);
    return { state: this.getState(), reward, done };
  }
}
