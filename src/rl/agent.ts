export interface AgentOptions {
  epsilon?: number;
  gamma?: number;
  learningRate?: number;
  epsilonDecay?: number;
  minEpsilon?: number;
}

export class RLAgent {
  initialEpsilon: number;
  epsilon: number;
  gamma: number;
  learningRate: number;
  epsilonDecay: number;
  minEpsilon: number;
  qTable: Map<string, Float32Array>;

  constructor(options: AgentOptions = {}) {
    this.initialEpsilon = options.epsilon ?? 0.1;
    this.epsilon = this.initialEpsilon;
    this.gamma = options.gamma ?? 0.95;
    this.learningRate = options.learningRate ?? 0.1;
    this.epsilonDecay = options.epsilonDecay ?? 0.99;
    this.minEpsilon = options.minEpsilon ?? 0.01;
    this.qTable = new Map();
  }

  protected _key(state: Float32Array): string {
    return Array.from(state).join(',');
  }

  protected _ensure(state: Float32Array): Float32Array {
    const key = this._key(state);
    if (!this.qTable.has(key)) {
      this.qTable.set(key, new Float32Array(4));
    }
    return this.qTable.get(key)!;
  }

  /** Choose an action using epsilon-greedy policy. */
  act(state: Float32Array): number {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 4);
    }
    const qVals = this._ensure(state);
    let best = 0;
    for (let i = 1; i < qVals.length; i++) {
      if (qVals[i] > qVals[best]) best = i;
    }
    return best;
  }

  /** Reduce exploration rate after learning. */
  decayEpsilon(): void {
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  /** Perform tabular Q-learning update. */
  learn(
    state: Float32Array,
    action: number,
    reward: number,
    nextState: Float32Array,
    done: boolean
  ): void | Promise<void> {
    const qVals = this._ensure(state);
    const nextQ = this._ensure(nextState);
    const maxNext = done ? 0 : Math.max(...nextQ);
    qVals[action] +=
      this.learningRate * (reward + this.gamma * maxNext - qVals[action]);
    this.decayEpsilon();
  }

  /** Reset agent to initial state. */
  reset(): void {
    this.epsilon = this.initialEpsilon;
    this.qTable.clear();
  }

  /** Serialize agent state to a plain object. */
  toJSON(): Record<string, unknown> {
    const table = Object.fromEntries(
      Array.from(this.qTable.entries()).map(([k, v]) => [k, Array.from(v)])
    );
    return {
      epsilon: this.epsilon,
      gamma: this.gamma,
      learningRate: this.learningRate,
      epsilonDecay: this.epsilonDecay,
      minEpsilon: this.minEpsilon,
      qTable: table
    };
  }

  /** Recreate an agent from serialized data. */
  static fromJSON(data: any): RLAgent {
    const agent = new RLAgent({
      epsilon: data.epsilon,
      gamma: data.gamma,
      learningRate: data.learningRate,
      epsilonDecay: data.epsilonDecay,
      minEpsilon: data.minEpsilon
    });
    for (const [k, v] of Object.entries(data.qTable || {})) {
      agent.qTable.set(k, new Float32Array(v as number[]));
    }
    return agent;
  }
}
