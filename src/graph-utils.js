/**
 * Graph utilities for the Oblix neural network playground.
 * 
 * Following Japanese-Level Code Quality principles:
 * - Monozukuri: Each function has a single, clear responsibility
 * - Kaizen: Extracted from the massive main.js function
 * - Wabi-sabi: Simple, focused functions that solve today's need
 */

/**
 * Draws the loss graph on the provided canvas context.
 * Business rule: Loss visualization helps users understand training progress.
 * 
 * @param {CanvasRenderingContext2D} ctx - Canvas context for drawing
 * @param {Array<number>} lossHistory - Array of loss values to plot
 * @param {number} canvasWidth - Width of the canvas
 * @param {number} canvasHeight - Height of the canvas
 */
export function drawLossGraph(ctx, lossHistory, canvasWidth, canvasHeight) {
  if (!ctx || !lossHistory || lossHistory.length === 0) {
    return;
  }

  // Clear canvas
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  // Set up plotting function
  const plot = (plotCtx, points, color) => {
    if (points.length < 2) return;
    
    plotCtx.strokeStyle = color;
    plotCtx.lineWidth = 2;
    plotCtx.beginPath();
    
    const xStep = canvasWidth / (points.length - 1);
    points.forEach((point, index) => {
      const x = index * xStep;
      const y = canvasHeight - (point * canvasHeight);
      if (index === 0) {
        plotCtx.moveTo(x, y);
      } else {
        plotCtx.lineTo(x, y);
      }
    });
    
    plotCtx.stroke();
  };

  // Normalize loss values to 0-1 range
  const maxLoss = Math.max(...lossHistory);
  const minLoss = Math.min(...lossHistory);
  const lossRange = maxLoss - minLoss;
  
  const normalizedLoss = lossHistory.map(loss => {
    if (lossRange === 0) return 0.5;
    return (loss - minLoss) / lossRange;
  });

  // Draw grid
  ctx.strokeStyle = '#ddd';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const y = (i / 10) * canvasHeight;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasWidth, y);
    ctx.stroke();
  }

  // Draw loss curve
  plot(ctx, normalizedLoss, '#007acc');
}

/**
 * Draws the neural network visualization on the provided canvas context.
 * Business rule: Network visualization helps users understand model architecture.
 * 
 * @param {CanvasRenderingContext2D} ctx - Canvas context for drawing
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Object} network - Neural network object with layers and activations
 * @param {number} containerWidth - Width of the container
 * @param {number} containerHeight - Height of the container
 */
export function drawNetwork(ctx, canvas, network, containerWidth, containerHeight) {
  if (!ctx || !canvas) return;
  
  const networkContainer = canvas.parentElement;
  if (!networkContainer) return;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  const hasModel = network.lastActivations &&
    network.lastActivations.length > 0 &&
    network.layers &&
    network.layers.length > 0;

  if (!hasModel) {
    ctx.fillStyle = '#555';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    if (canvas.width !== containerWidth) {
      canvas.width = containerWidth;
    }
    if (canvas.height !== containerHeight) {
      canvas.height = containerHeight;
    }
    
    ctx.fillText(
      'Train/Predict to visualize',
      containerWidth / 2,
      containerHeight / 2
    );
    return;
  }

  // Drawing constants
  const pad = 35;
  const maxNodes = 20;
  const nodeRadiusBase = 2;
  const nodeRadiusScale = 3;
  const connectionBaseOpacity = 0.02;
  const connectionMaxOpacity = 0.85;
  const connectionWeightScale = 2;
  const ellipseOffset = 10;
  const labelOffset = 20;
  const labelFont = '10px monospace';
  const labelColor = '#aaa';

  const numVisualLayers = network.lastActivations.length;
  const baseSpacing = 150;
  const minLayerSpacing = Math.max(
    120,
    Math.min(baseSpacing, containerWidth / (numVisualLayers > 1 ? numVisualLayers : 1))
  );

  const requiredWidth = numVisualLayers <= 1
    ? containerWidth
    : pad * 2 + (numVisualLayers - 1) * minLayerSpacing;

  const canvasDrawWidth = Math.max(containerWidth, requiredWidth * 1.2);
  canvas.width = canvasDrawWidth;
  canvas.height = containerHeight;

  const drawAreaWidth = canvasDrawWidth - pad * 2;
  const drawAreaHeight = containerHeight - pad * 2;
  
  const layerXPositions = Array.from(
    { length: numVisualLayers },
    (_, index) => pad + (numVisualLayers === 1
      ? drawAreaWidth / 2
      : (drawAreaWidth * index) / (numVisualLayers - 1))
  );

  // Draw layers and nodes
  const layerPositions = [];
  network.lastActivations.forEach((activation, layerIndex) => {
    if (!activation || typeof activation.length !== 'number') {
      if (network.debug) {
        console.warn(
          `drawNetwork L${layerIndex}: Activation data is not array-like.`,
          activation
        );
      }
      layerPositions.push([]);
      return;
    }

    const layerNodes = [];
    const numNodes = activation.length;
    const displayNodes = Math.min(numNodes, maxNodes);
    const layerX = layerXPositions[layerIndex];
    
    for (let nodeIndex = 0; nodeIndex < displayNodes; nodeIndex++) {
      const originalIndex = numNodes <= maxNodes 
        ? nodeIndex 
        : Math.floor((nodeIndex * numNodes) / displayNodes);
      const nodeValue = activation[originalIndex];
      const nodeY = pad + (displayNodes === 1
        ? drawAreaHeight / 2
        : (drawAreaHeight * nodeIndex) / (displayNodes - 1));

      layerNodes.push({
        x: layerX,
        y: nodeY,
        value: nodeValue,
        originalIndex: originalIndex
      });
    }
    
    layerPositions.push(layerNodes);
  });

  // Draw connections between layers
  for (let layerIndex = 0; layerIndex < layerPositions.length - 1; layerIndex++) {
    const currentLayer = layerPositions[layerIndex];
    const nextLayer = layerPositions[layerIndex + 1];
    
    if (!currentLayer || !nextLayer) continue;
    
    currentLayer.forEach(sourceNode => {
      nextLayer.forEach(targetNode => {
        const connectionOpacity = Math.min(
          connectionMaxOpacity,
          Math.max(connectionBaseOpacity, Math.abs(sourceNode.value) * connectionWeightScale)
        );
        
        ctx.strokeStyle = `rgba(100, 150, 255, ${connectionOpacity})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sourceNode.x, sourceNode.y);
        ctx.lineTo(targetNode.x, targetNode.y);
        ctx.stroke();
      });
    });
  }

  // Draw nodes
  layerPositions.forEach(layerNodes => {
    layerNodes.forEach(node => {
      const nodeRadius = nodeRadiusBase + Math.abs(node.value) * nodeRadiusScale;
      const nodeColor = node.value >= 0 ? '#4CAF50' : '#F44336';
      
      ctx.fillStyle = nodeColor;
      ctx.beginPath();
      ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw node label
      ctx.fillStyle = labelColor;
      ctx.font = labelFont;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        node.originalIndex.toString(),
        node.x,
        node.y + labelOffset
      );
    });
  });

  // Draw layer labels
  layerXPositions.forEach((layerX, layerIndex) => {
    ctx.fillStyle = labelColor;
    ctx.font = labelFont;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(
      `L${layerIndex}`,
      layerX,
      pad - labelOffset
    );
  });
}