export class LiveChart {
  constructor(canvas, windowSize = 0) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.movingAvgWindow = windowSize;
    this.rewards = [];
    this.epsilons = [];
    this.avgRewards = [];
    this.resize = this.resize.bind(this);
    this.resize();
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', this.resize);
    }
  }

  push(reward, epsilon) {
    this.rewards.push(reward);
    this.epsilons.push(epsilon);
    if (this.movingAvgWindow > 1) {
      const start = Math.max(0, this.rewards.length - this.movingAvgWindow);
      const slice = this.rewards.slice(start);
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
      this.avgRewards.push(avg);
    }
    this.draw();
  }

  resize() {
    if (this.canvas.parentElement) {
      this.canvas.width = this.canvas.parentElement.clientWidth;
    }
    this.draw();
  }

  draw() {
    const { ctx, canvas, rewards, epsilons, avgRewards, movingAvgWindow } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (rewards.length === 0) return;
    ctx.beginPath();
    const gridLines = 5;
    for (let i = 1; i < gridLines; i++) {
      const y = (canvas.height / gridLines) * i;
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      const x = (canvas.width / gridLines) * i;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
    }
    ctx.strokeStyle = '#ccc';
    ctx.stroke();

    const maxReward = Math.max(...rewards);
    const minReward = Math.min(...rewards);
    const rewardRange = maxReward - minReward || 1;
    const stepX = canvas.width / Math.max(rewards.length - 1, 1);

    ctx.beginPath();
    ctx.moveTo(0, canvas.height);
    ctx.lineTo(canvas.width, canvas.height);
    ctx.moveTo(0, 0);
    ctx.lineTo(0, canvas.height);

    const xTicks = 5;
    const xStepEpisodes = Math.max(1, Math.floor((rewards.length - 1) / xTicks));
    for (let i = 0; i <= rewards.length - 1; i += xStepEpisodes) {
      const x = i * stepX;
      ctx.moveTo(x, canvas.height);
      ctx.lineTo(x, canvas.height - 5);
    }

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const value = minReward + (rewardRange * i) / yTicks;
      const y = canvas.height - ((value - minReward) / rewardRange) * canvas.height;
      ctx.moveTo(0, y);
      ctx.lineTo(5, y);
    }
    ctx.strokeStyle = '#999';
    ctx.stroke();

    ctx.fillStyle = '#ccc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    for (let i = 0; i <= rewards.length - 1; i += xStepEpisodes) {
      const x = i * stepX;
      ctx.fillText(String(i), x, canvas.height - 7);
    }

    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= yTicks; i++) {
      const value = minReward + (rewardRange * i) / yTicks;
      const y = canvas.height - ((value - minReward) / rewardRange) * canvas.height;
      ctx.fillText(value.toFixed(2), 10, y);
    }

    ctx.beginPath();
    rewards.forEach((r, i) => {
      const x = i * stepX;
      const y = canvas.height - ((r - minReward) / rewardRange) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#4caf50';
    ctx.stroke();

    if (movingAvgWindow > 1 && avgRewards.length > 0) {
      ctx.beginPath();
      avgRewards.forEach((r, i) => {
        const x = i * stepX;
        const y = canvas.height - ((r - minReward) / rewardRange) * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = '#ff9800';
      ctx.stroke();
    }

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

    ctx.font = '12px sans-serif';
    const legendX = 10;
    let legendY = 10;

    ctx.fillStyle = '#4caf50';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Reward', legendX + 15, legendY + 10);

    if (movingAvgWindow > 1) {
      legendY += 20;
      ctx.fillStyle = '#ff9800';
      ctx.fillRect(legendX, legendY, 10, 10);
      ctx.fillStyle = '#ccc';
      ctx.fillText('Avg Reward', legendX + 15, legendY + 10);
    }

    legendY += 20;
    ctx.fillStyle = '#2196f3';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Epsilon', legendX + 15, legendY + 10);
  }
}
