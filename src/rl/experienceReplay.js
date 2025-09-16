export class ExperienceReplay {
  constructor(capacity = 1000, alpha = 0.6, beta = 0.4, betaIncrement = 0) {
    this.capacity = capacity;
    this.alpha = alpha;
    this.beta = beta;
    this.betaIncrement = betaIncrement;
    this.buffer = [];
    this.priorities = [];
    this.position = 0;
  }

  get size() {
    return this.buffer.length;
  }

  clear() {
    this.buffer.length = 0;
    this.priorities.length = 0;
    this.position = 0;
  }

  add(transition, priority = 1) {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(transition);
      this.priorities.push(priority);
    } else {
      this.buffer[this.position] = transition;
      this.priorities[this.position] = priority;
    }
    this.position = (this.position + 1) % this.capacity;
  }

  sample(count, strategy = 'uniform') {
    if (this.buffer.length === 0 || count <= 0) return [];
    return strategy === 'priority'
      ? this._samplePriority(count)
      : this._sampleUniform(count);
  }

  _randomIndex() {
    return Math.floor(Math.random() * this.buffer.length);
  }

  _sampleUniform(count) {
    const batch = [];
    for (let i = 0; i < count; i++) {
      const idx = this._randomIndex();
      batch.push({ index: idx, weight: 1, ...this.buffer[idx] });
    }
    return batch;
  }

  _annealBeta() {
    if (this.betaIncrement > 0 && this.beta < 1) {
      this.beta = Math.min(1, this.beta + this.betaIncrement);
    }
  }

  _samplePriority(count) {
    const size = this.buffer.length;
    const scaled = this.priorities
      .slice(0, size)
      .map(p => Math.pow(Math.max(p, 0), this.alpha));
    const total = scaled.reduce((a, b) => a + b, 0);
    if (total === 0) {
      const batch = this._sampleUniform(count);
      this._annealBeta();
      return batch;
    }
    const probabilities = scaled.map(w => w / total);
    const weighted = probabilities.map(p =>
      p > 0 ? Math.pow(size * p, -this.beta) : 0
    );
    const maxWeight = weighted.reduce((max, val) => (val > max ? val : max), 0) || 1;
    const normalizedWeights = weighted.map(w => (w === 0 ? 0 : w / maxWeight));
    const batch = [];
    for (let i = 0; i < count; i++) {
      const r = Math.random();
      let acc = 0;
      let selected = size - 1;
      for (let j = 0; j < size; j++) {
        acc += probabilities[j];
        if (r <= acc) {
          selected = j;
          break;
        }
      }
      batch.push({ index: selected, weight: normalizedWeights[selected], ...this.buffer[selected] });
    }
    this._annealBeta();
    return batch;
  }

  updatePriority(index, priority) {
    if (index >= 0 && index < this.priorities.length) {
      this.priorities[index] = priority;
    }
  }
}
