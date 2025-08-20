export class LiveChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.rewards = [];
    this.epsilons = [];
  }

  push(reward, epsilon) {
    this.rewards.push(reward);
    this.epsilons.push(epsilon);
    this.draw();
  }

  draw() {
    const { ctx, canvas, rewards, epsilons } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (rewards.length === 0) return;
    const maxReward = Math.max(...rewards);
    const minReward = Math.min(...rewards);
    const rewardRange = maxReward - minReward || 1;
    const stepX = canvas.width / Math.max(rewards.length - 1, 1);
    ctx.beginPath();
    rewards.forEach((r, i) => {
      const x = i * stepX;
      const y = canvas.height - ((r - minReward) / rewardRange) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#4caf50';
    ctx.stroke();

    const epsRange = 1;
    ctx.beginPath();
    epsilons.forEach((e, i) => {
      const x = i * stepX;
      const y = canvas.height - (e / epsRange) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#2196f3';
    ctx.stroke();
  }
}
