export class GridWorldEnvironment {
  constructor(size = 5, obstacles = []) {
    this.size = size;
    this.obstacles = [];
    this.obstacleSet = new Set();
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
      return { state: this.getState(), reward: -0.1, done: false };
    }
    this.agentPos = { x: newX, y: newY };
    const done =
      this.agentPos.x === this.size - 1 && this.agentPos.y === this.size - 1;
    const reward = done ? 1 : -0.01;
    return { state: this.getState(), reward, done };
  }
}
