import { RLAgent, AgentOptions } from './agent.js';

interface DynaQOptions extends AgentOptions {
  planningSteps?: number;
}

interface ModelEntry {
  nextState: Float32Array;
  reward: number;
  done: boolean;
}

export class DynaQAgent extends RLAgent {
  planningSteps: number;
  model: Map<string, ModelEntry>;
  stateActions: { state: Float32Array; action: number }[];

  constructor(options: DynaQOptions = {}) {
    super(options);
    this.planningSteps = options.planningSteps ?? 5;
    this.model = new Map();
    this.stateActions = [];
  }

  private _saKey(state: Float32Array, action: number): string {
    return `${this._key(state)}|${action}`;
  }

  learn(
    state: Float32Array,
    action: number,
    reward: number,
    nextState: Float32Array,
    done: boolean
  ): void | Promise<void> {
    const s = new Float32Array(state);
    const ns = new Float32Array(nextState);
    const key = this._saKey(s, action);
    if (!this.model.has(key)) {
      this.stateActions.push({ state: s, action });
    }
    this.model.set(key, { nextState: ns, reward, done });

    const qVals = this._ensure(s);
    const nextQ = this._ensure(ns);
    const maxNext = done ? 0 : Math.max(...nextQ);
    qVals[action] += this.learningRate * (reward + this.gamma * maxNext - qVals[action]);

    const total = this.stateActions.length;
    for (let i = 0; i < this.planningSteps && total > 0; i++) {
      const idx = Math.floor(Math.random() * total);
      const { state: ps, action: pa } = this.stateActions[idx];
      const modelKey = this._saKey(ps, pa);
      const modelEntry = this.model.get(modelKey)!;
      const pns = modelEntry.nextState;
      const pr = modelEntry.reward;
      const pd = modelEntry.done;
      const pq = this._ensure(ps);
      const pnextQ = this._ensure(pns);
      const pmaxNext = pd ? 0 : Math.max(...pnextQ);
      pq[pa] += this.learningRate * (pr + this.gamma * pmaxNext - pq[pa]);
    }

    this.decayEpsilon();
  }
}
