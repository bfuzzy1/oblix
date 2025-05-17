export function drawNetwork(nn, canvas, ctx) {
  if (!ctx || !canvas) return;
  const container = canvas.parentElement;
  if (!container) return;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const hasModel =
    nn.lastActivations && nn.lastActivations.length > 0 && nn.layers && nn.layers.length > 0;
  if (!hasModel) {
    ctx.fillStyle = "#555";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (canvas.width !== containerWidth) canvas.width = containerWidth;
    if (canvas.height !== containerHeight) canvas.height = containerHeight;
    ctx.fillText("Train/Predict to visualize", containerWidth / 2, containerHeight / 2);
    return;
  }
  const pad = 35,
    maxNds = 20,
    nRBase = 2,
    nRScale = 3,
    cBOp = 0.02,
    cMOp = 0.85,
    cWScale = 2,
    ellOff = 10,
    lblOff = 20,
    lblFnt = "10px monospace",
    lblClr = "#aaa";
  const nVizLyrs = nn.lastActivations.length;
  const baseSpacing = 150;
  const minLayerSpacing = Math.max(120, Math.min(baseSpacing, containerWidth / (nVizLyrs > 1 ? nVizLyrs : 1)));
  const requiredWidth =
    nVizLyrs <= 1 ? containerWidth : pad * 2 + (nVizLyrs - 1) * minLayerSpacing;
  const canvasDrawWidth = Math.max(containerWidth, requiredWidth * 1.2);
  canvas.width = canvasDrawWidth;
  canvas.height = containerHeight;
  const drawAreaWidth = canvasDrawWidth - pad * 2;
  const drawAreaHeight = containerHeight - pad * 2;
  const layerXs = Array.from({ length: nVizLyrs }, (_, i) =>
    pad + (nVizLyrs === 1 ? drawAreaWidth / 2 : (drawAreaWidth * i) / (nVizLyrs - 1)),
  );
  const layerPos = [];
  nn.lastActivations.forEach((act, lIdx) => {
    if (!act || typeof act.length !== "number") {
      if (nn.debug) console.warn(`drawNetwork L${lIdx}: Activation data is not array-like.`, act);
      layerPos.push([]);
      return;
    }
    const lNodes = [];
    const nNodes = act.length;
    const dNodes = Math.min(nNodes, maxNds);
    const lX = layerXs[lIdx];
    for (let j = 0; j < dNodes; j++) {
      const origIdx = nNodes <= maxNds ? j : Math.floor((j * nNodes) / dNodes);
      const nodeVal = act[origIdx];
      const nY = pad + (dNodes === 1 ? drawAreaHeight / 2 : (drawAreaHeight * j) / (dNodes - 1));
      lNodes.push({ x: lX, y: nY, value: typeof nodeVal === "number" && isFinite(nodeVal) ? nodeVal : 0 });
    }
    if (nNodes > maxNds) {
      lNodes.push({ x: lX, y: pad + drawAreaHeight + ellOff, value: 0, isEllipsis: true, originalCount: nNodes });
    }
    layerPos.push(lNodes);
  });
  ctx.lineWidth = 1;
  for (let i = 0; i < nVizLyrs - 1; i++) {
    const curNodes = layerPos[i].filter((n) => !n.isEllipsis);
    const nextNodes = layerPos[i + 1].filter((n) => !n.isEllipsis);
    const cfg = nn.layers[i];
    if (!cfg) continue;
    const isDenseW = cfg.type === "dense" && nn.weights?.[i] instanceof Float32Array;
    const w = isDenseW ? nn.weights[i] : null;
    const inputSizeForLayer = cfg.inputSize;
    for (let j = 0; j < curNodes.length; j++) {
      for (let k = 0; k < nextNodes.length; k++) {
        let op = 0.1,
          col = "100,100,100",
          lw = 0.5;
        let lineDash = [];
        let weight = null;
        if (isDenseW && w && typeof inputSizeForLayer === "number" && inputSizeForLayer > 0) {
          const weightIndex = k * inputSizeForLayer + j;
          if (weightIndex >= 0 && weightIndex < w.length) {
            weight = w[weightIndex];
          } else {
            if (nn.debug)
              console.warn(
                `drawNetwork L${i} Dense: Invalid weight index ${weightIndex}(nextIdx=${k}, currIdx=${j}, inSize=${inputSizeForLayer}, wLen=${w.length})`,
              );
            weight = null;
          }
        }
        if (isDenseW && weight !== null && typeof weight === "number" && typeof curNodes[j].value === "number") {
          const wMag = Math.tanh(Math.abs(weight));
          const aMag = Math.tanh(Math.abs(curNodes[j].value));
          const combSig = wMag * 0.8 + aMag * 0.2;
          op = Math.min(Math.max(cBOp + combSig * (cMOp - cBOp), cBOp), cMOp);
          col = weight >= 0 ? "255,255,255" : "180,180,255";
          lw = Math.min(Math.max(0.5, op * cWScale), 2);
          lineDash = [];
        } else if (cfg.type === "attention") {
          op = 0.5;
          lw = 1.2;
          col = "0,200,200";
          lineDash = [];
        } else if (cfg.type === "layernorm") {
          op = 0.4;
          lw = 0.8;
          col = "255,255,0";
          lineDash = [2, 2];
        } else if (cfg.type === "dropout") {
          op = 0.2;
          lw = 0.6;
          col = "255,165,0";
          lineDash = [3, 3];
        } else if (cfg.type === "softmax") {
          op = 0.4;
          lw = 1.0;
          col = "200,0,200";
          lineDash = [];
        } else if (!isDenseW) {
          op = 0.1;
          lw = 0.5;
          col = "100,100,100";
          lineDash = [];
        } else {
          op = 0.05;
          lw = 0.3;
          col = "80,80,80";
          lineDash = [1, 2];
        }
        ctx.strokeStyle = `rgba(${col},${op})`;
        ctx.lineWidth = lw;
        ctx.setLineDash(lineDash || []);
        ctx.beginPath();
        ctx.moveTo(curNodes[j].x, curNodes[j].y);
        ctx.lineTo(nextNodes[k].x, nextNodes[k].y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }
  ctx.textAlign = "center";
  if (layerXs.length > 2) {
    const firstHiddenLayerX = layerXs[1];
    const lastHiddenLayerX = layerXs[layerXs.length - 2];
    const centerX = (firstHiddenLayerX + lastHiddenLayerX) / 2;
    ctx.fillStyle = lblClr;
    ctx.font = lblFnt;
    ctx.textBaseline = "bottom";
    ctx.fillText("Layers", centerX, pad - lblOff / 2);
  }
  layerPos.forEach((lNodes, lIdx) => {
    ctx.fillStyle = lblClr;
    ctx.font = lblFnt;
    ctx.textBaseline = "bottom";
    if (lIdx === 0) {
      ctx.fillText("Input", layerXs[lIdx], pad - lblOff / 2);
    } else if (lIdx === layerPos.length - 1) {
      ctx.fillText("Output", layerXs[lIdx], pad - lblOff / 2);
    }
    lNodes.forEach((n) => {
      if (n.isEllipsis) {
        ctx.fillStyle = "#777";
        ctx.font = "10px monospace";
        ctx.textBaseline = "top";
        ctx.fillText(`(${n.originalCount} nodes)`, n.x, n.y);
      } else {
        const actStr = Math.tanh(Math.abs(n.value));
        const r = nRBase + actStr * nRScale;
        const op = 0.3 + actStr * 0.7;
        const col = n.value >= 0 ? "255,255,255" : "200,200,255";
        ctx.fillStyle = `rgba(${col},${op})`;
        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    });
  });
}
