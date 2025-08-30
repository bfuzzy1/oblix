export class ActorCriticAgent {
  constructor(options = {}) {
    this.gamma = options.gamma ?? 0.95;
    this.alphaCritic = options.alphaCritic ?? 0.1;
    this.alphaActor = options.alphaActor ?? 0.1;
    this.temperature = options.temperature ?? 1;
    this.policyTable = new Map();
    this.valueTable = new Map();
  }

  _key(state) {
    return Array.from(state).join(',');
  }

  _ensurePolicy(state) {
    const key = this._key(state);
    if (!this.policyTable.has(key)) {
      this.policyTable.set(key, new Float32Array(4));
    }
    return this.policyTable.get(key);
  }

  _ensureValue(state) {
    const key = this._key(state);
    if (!this.valueTable.has(key)) {
      this.valueTable.set(key, new Float32Array(4));
    }
    return this.valueTable.get(key);
  }

  _softmax(prefs) {
    const max = Math.max(...prefs);
    const exps = prefs.map(v => Math.exp((v - max) / this.temperature));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  policyProbs(state) {
    const prefs = Array.from(this._ensurePolicy(state));
    return this._softmax(prefs);
  }

  act(state) {
    const probs = this.policyProbs(state);
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  learn(state, action, reward, nextState, done) {
    const values = this._ensureValue(state);
    const nextValues = this._ensureValue(nextState);
    const prefs = this._ensurePolicy(state);
    const probs = this.policyProbs(state);
    const current = values[action];
    const maxNext = done ? 0 : Math.max(...nextValues);
    const tdError = reward + this.gamma * maxNext - current;
    values[action] += this.alphaCritic * tdError;
    for (let i = 0; i < prefs.length; i++) {
      const grad = (i === action ? 1 - probs[i] : -probs[i]);
      prefs[i] += this.alphaActor * tdError * grad;
    }
  }

  reset() {
    this.policyTable.clear();
    this.valueTable.clear();
  }
}
