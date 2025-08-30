export class ExperienceReplay {
  constructor(capacity = 1000, alpha = 0.6, beta = 0.4) {
    this.capacity = capacity;
    this.alpha = alpha;
    this.beta = beta;
    this.buffer = [];
    this.priorities = [];
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
    if (this.buffer.length === 0) return [];
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
      batch.push({ index: idx, ...this.buffer[idx] });
    }
    return batch;
  }

  _samplePriority(count) {
    const weights = this.priorities
      .slice(0, this.buffer.length)
      .map(p => Math.pow(p, this.alpha));
    const total = weights.reduce((a, b) => a + b, 0);
    if (total === 0) {
      return this._sampleUniform(count);
    }
    const batch = [];
    for (let i = 0; i < count; i++) {
      const r = Math.random() * total;
      let acc = 0;
      for (let j = 0; j < this.buffer.length; j++) {
        acc += weights[j];
        if (r <= acc) {
          batch.push({ index: j, ...this.buffer[j] });
          break;
        }
      }
    }
    return batch;
  }

  updatePriority(index, priority) {
    if (index >= 0 && index < this.priorities.length) {
      this.priorities[index] = priority;
    }
  }
}
