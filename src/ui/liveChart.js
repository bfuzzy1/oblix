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

    ctx.font = '12px sans-serif';
    const legendX = 10;
    let legendY = 10;

    ctx.fillStyle = '#4caf50';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Reward', legendX + 15, legendY + 10);

    legendY += 20;
    ctx.fillStyle = '#2196f3';
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Epsilon', legendX + 15, legendY + 10);
  }
}
