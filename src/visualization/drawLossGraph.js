export function drawLossGraph(ctx, canvas, history) {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (history.length < 2) return;
  const trainL = history.map((h) => h.train).filter((l) => l !== null && isFinite(l));
  const testL = history.map((h) => h.test).filter((l) => l !== null && isFinite(l));
  let maxL = 0.1;
  if (trainL.length > 0) maxL = Math.max(maxL, ...trainL);
  if (testL.length > 0) maxL = Math.max(maxL, ...testL);
  maxL = Math.max(maxL, 0.1);
  const W = canvas.width,
    H = canvas.height,
    nPts = history.length,
    pH = H * 0.9,
    yOff = H * 0.05;
  const plot = (pts, c) => {
    ctx.strokeStyle = c;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let first = true;
    pts.forEach((p, i) => {
      if (p !== null && isFinite(p)) {
        const x = (i / Math.max(1, nPts - 1)) * W;
        const y = H - (p / maxL) * pH - yOff;
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      } else {
        first = true;
      }
    });
    ctx.stroke();
  };
  const trainC = getComputedStyle(document.body).getPropertyValue("--text")?.trim() || "#fff";
  plot(history.map((h) => h.train), trainC);
  plot(history.map((h) => h.test), "#87CEEB");
}
