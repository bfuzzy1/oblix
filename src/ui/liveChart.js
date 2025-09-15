export class LiveChart {
  constructor(canvas, windowSize = 10) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.windowSize = windowSize;
    this.rewards = [];
    this.epsilons = [];
    this.avgRewards = [];
  }

  push(reward, epsilon) {
    this.rewards.push(reward);
    this.epsilons.push(epsilon);
    const start = Math.max(0, this.rewards.length - this.windowSize);
    const slice = this.rewards.slice(start);
    const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
    this.avgRewards.push(avg);
    this.draw();
  }

  draw() {
    const { ctx, canvas, rewards, epsilons, avgRewards } = this;
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
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
    ctx.stroke();
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
    ctx.strokeStyle = '#5eead4';
    ctx.stroke();

    ctx.beginPath();
    avgRewards.forEach((r, i) => {
      const x = i * stepX;
      const y = canvas.height - ((r - minReward) / rewardRange) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#fbbf24';
    ctx.stroke();

    const epsRange = 1;
    ctx.beginPath();
    epsilons.forEach((e, i) => {
      const x = i * stepX;
      const y = canvas.height - (e / epsRange) * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#38bdf8';
    ctx.stroke();

    ctx.font = "12px 'Inter', sans-serif";
    const legendX = 10;
    let legendY = 10;

    ctx.fillStyle = '#5eead4';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText('Reward', legendX + 15, legendY + 10);

    legendY += 20;
    ctx.fillStyle = '#fbbf24';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText('Avg Reward', legendX + 15, legendY + 10);

    legendY += 20;
    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText('Epsilon', legendX + 15, legendY + 10);
  }
}
