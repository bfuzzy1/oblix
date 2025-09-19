export const DEFAULT_REWARD_CONFIG = Object.freeze({
  stepPenalty: -0.01,
  obstaclePenalty: -0.1,
  goalReward: 1
});

export const GRID_ACTIONS = Object.freeze([0, 1, 2, 3]);

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

  isGoalPosition(x, y) {
    const goal = this.getGoalPosition();
    return x === goal.x && y === goal.y;
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

  getPositionFromState(state) {
    if (!state) {
      throw new Error('State is required to derive a position');
    }
    if (ArrayBuffer.isView(state) || Array.isArray(state)) {
      const [sx, sy] = state;
      return { x: Math.trunc(Number(sx)), y: Math.trunc(Number(sy)) };
    }
    if (typeof state === 'object' && state !== null) {
      const { x, y } = state;
      if (Number.isFinite(x) && Number.isFinite(y)) {
        return { x: Math.trunc(x), y: Math.trunc(y) };
      }
    }
    throw new Error('Unsupported state representation');
  }

  createStateFromPosition(x, y) {
    return new Float32Array([x, y]);
  }

  enumerateCells() {
    const cells = [];
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        if (this.isObstacle(x, y)) continue;
        cells.push({ x, y });
      }
    }
    return cells;
  }

  enumerateStates() {
    return this.enumerateCells().map(cell => this.createStateFromPosition(cell.x, cell.y));
  }

  getAvailableActions() {
    return GRID_ACTIONS;
  }

  isTerminalState(state) {
    const { x, y } = this.getPositionFromState(state);
    return this.isGoalPosition(x, y);
  }

  _simulateAction(position, action) {
    let newX = position.x;
    let newY = position.y;
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
    return { x: newX, y: newY };
  }

  _resolveTransition(position, action) {
    const candidate = this._simulateAction(position, action);
    if (this.isObstacle(candidate.x, candidate.y)) {
      return {
        nextPosition: { x: position.x, y: position.y },
        reward: this.obstaclePenalty,
        done: false
      };
    }
    const done = this.isGoalPosition(candidate.x, candidate.y);
    const reward = this.calculateReward(candidate.x, candidate.y, done);
    return {
      nextPosition: candidate,
      reward,
      done
    };
  }

  getTransition(state, action) {
    const position = this.getPositionFromState(state);
    const { nextPosition, reward, done } = this._resolveTransition(position, action);
    return {
      state: this.createStateFromPosition(nextPosition.x, nextPosition.y),
      reward,
      done
    };
  }

  /**
   * Step the environment with an action.
   * @param action 0:up,1:down,2:left,3:right
   */
  step(action) {
    const { nextPosition, reward, done } = this._resolveTransition(this.agentPos, action);
    this.agentPos = { x: nextPosition.x, y: nextPosition.y };
    return { state: this.getState(), reward, done };
  }
}
