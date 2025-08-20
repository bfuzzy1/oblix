export type EnvironmentState = Float32Array;
export interface StepResult {
  state: EnvironmentState;
  reward: number;
  done: boolean;
}
export declare class GridWorldEnvironment {
  size: number;
  obstacles: {
  x: number;
  y: number;
  }[];
  obstacleSet: Set<string>;
  agentPos: {
  x: number;
  y: number;
  };
  constructor(size?: number, obstacles?: {
  x: number;
  y: number;
  }[]);
  /** Reset the environment to the starting state. */
  reset(): EnvironmentState;
  /** Convert the current agent position to state Float32Array. */
  getState(): EnvironmentState;
  isObstacle(x: number, y: number): boolean;
  toggleObstacle(x: number, y: number): void;
  /**
   * Step the environment with an action.
   * @param action 0:up,1:down,2:left,3:right
   */
  step(action: number): StepResult;
}
