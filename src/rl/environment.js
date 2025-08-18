export class GridWorldEnvironment {
  constructor(size = 5) {
    this.size = size;
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

  /**
   * Step the environment with an action.
   * @param {number} action 0:up,1:down,2:left,3:right
   * @returns {{state:Float32Array,reward:number,done:boolean}}
   */
  step(action) {
    switch (action) {
      case 0:
        if (this.agentPos.y > 0) this.agentPos.y -= 1;
        break;
      case 1:
        if (this.agentPos.y < this.size - 1) this.agentPos.y += 1;
        break;
      case 2:
        if (this.agentPos.x > 0) this.agentPos.x -= 1;
        break;
      case 3:
        if (this.agentPos.x < this.size - 1) this.agentPos.x += 1;
        break;
    }
    const done =
      this.agentPos.x === this.size - 1 && this.agentPos.y === this.size - 1;
    const reward = done ? 1 : -0.01;
    return { state: this.getState(), reward, done };
  }
}
