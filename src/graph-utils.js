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
 * Validates network data for visualization.
 * Business rule: Network validation prevents runtime errors.
 *
 * @param {Object} network - Neural network object
 * @returns {boolean} True if network has valid data for visualization
 */
function validateNetworkForVisualization(network) {
  return network.lastActivations &&
    network.lastActivations.length > 0 &&
    network.layers &&
    network.layers.length > 0;
}

/**
 * Draws placeholder text when no network data is available.
 * Business rule: User feedback is important for empty states.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {number} containerWidth - Container width
 * @param {number} containerHeight - Container height
 */
function drawEmptyNetworkMessage(ctx, canvas, containerWidth, containerHeight) {
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
}

/**
 * Calculates drawing constants for network visualization.
 * Business rule: Consistent visual parameters improve user experience.
 *
 * @returns {Object} Drawing constants
 */
function getDrawingConstants() {
  return {
    pad: 35,
    maxNodes: 20,
    nodeRadiusBase: 2,
    nodeRadiusScale: 3,
    connectionBaseOpacity: 0.02,
    connectionMaxOpacity: 0.85,
    connectionWeightScale: 2,
    ellipseOffset: 10,
    labelOffset: 20,
    labelFont: '10px monospace',
    labelColor: '#aaa'
  };
}

/**
 * Calculates canvas dimensions and layer positions.
 * Business rule: Proper spacing ensures clear visualization.
 *
 * @param {number} numVisualLayers - Number of layers to visualize
 * @param {number} containerWidth - Container width
 * @param {number} containerHeight - Container height
 * @param {Object} constants - Drawing constants
 * @returns {Object} Canvas and positioning data
 */
function calculateCanvasDimensions(numVisualLayers, containerWidth, containerHeight, constants) {
  const { pad } = constants;
  const baseSpacing = 150;
  const minLayerSpacing = Math.max(
    120,
    Math.min(baseSpacing, containerWidth / (numVisualLayers > 1 ? numVisualLayers : 1))
  );

  const requiredWidth = numVisualLayers <= 1
    ? containerWidth
    : pad * 2 + (numVisualLayers - 1) * minLayerSpacing;

  const canvasDrawWidth = Math.max(containerWidth, requiredWidth * 1.2);
  const drawAreaWidth = canvasDrawWidth - pad * 2;
  const drawAreaHeight = containerHeight - pad * 2;
  
  const layerXPositions = Array.from(
    { length: numVisualLayers },
    (_, index) => pad + (numVisualLayers === 1
      ? drawAreaWidth / 2
      : (drawAreaWidth * index) / (numVisualLayers - 1))
  );

  return {
    canvasDrawWidth,
    canvasDrawHeight: containerHeight,
    drawAreaWidth,
    drawAreaHeight,
    layerXPositions
  };
}

/**
 * Draws a single node in the network visualization.
 * Business rule: Node appearance reflects activation values.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x - X position
 * @param {number} y - Y position
 * @param {number} value - Node activation value
 * @param {Object} constants - Drawing constants
 */
function drawNode(ctx, x, y, value, constants) {
  const { nodeRadiusBase, nodeRadiusScale } = constants;
  const radius = nodeRadiusBase + Math.abs(value) * nodeRadiusScale;
  const alpha = Math.min(0.9, 0.3 + Math.abs(value) * 0.6);
  
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = `rgba(100, 150, 255, ${alpha})`;
  ctx.fill();
}

/**
 * Draws connections between layers.
 * Business rule: Connection opacity reflects weight strength.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array} prevLayerNodes - Previous layer node positions
 * @param {Array} currentLayerNodes - Current layer node positions
 * @param {Object} constants - Drawing constants
 */
function drawConnections(ctx, prevLayerNodes, currentLayerNodes, constants) {
  const { connectionBaseOpacity, connectionMaxOpacity, connectionWeightScale } = constants;
  
  prevLayerNodes.forEach(prevNode => {
    currentLayerNodes.forEach(currentNode => {
      const opacity = connectionBaseOpacity + 
        (Math.abs(prevNode.value) + Math.abs(currentNode.value)) * 
        connectionMaxOpacity * 0.5;
      
      ctx.beginPath();
      ctx.moveTo(prevNode.x, prevNode.y);
      ctx.lineTo(currentNode.x, currentNode.y);
      ctx.strokeStyle = `rgba(150, 150, 150, ${opacity})`;
      ctx.lineWidth = connectionWeightScale * opacity;
      ctx.stroke();
    });
  });
}

/**
 * Draws layer labels.
 * Business rule: Clear labeling improves user understanding.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x - X position
 * @param {number} y - Y position
 * @param {string} label - Layer label
 * @param {Object} constants - Drawing constants
 */
function drawLayerLabel(ctx, x, y, label, constants) {
  const { labelFont, labelColor, labelOffset } = constants;
  
  ctx.font = labelFont;
  ctx.fillStyle = labelColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(label, x, y + labelOffset);
}

/**
 * Processes activation data for a layer.
 * Business rule: Data validation prevents visualization errors.
 *
 * @param {Array} activation - Layer activation data
 * @param {number} layerIndex - Layer index
 * @param {number} maxNodes - Maximum nodes to display
 * @param {number} layerX - X position of the layer
 * @param {number} pad - Padding
 * @param {number} drawAreaHeight - Drawing area height
 * @param {boolean} debug - Debug mode
 * @returns {Array} Processed node positions
 */
function processLayerActivations(activation, layerIndex, maxNodes, layerX, pad, drawAreaHeight, debug) {
  if (!activation || typeof activation.length !== 'number') {
    if (debug) {
      console.warn(
        `drawNetwork L${layerIndex}: Activation data is not array-like.`,
        activation
      );
    }
    return [];
  }

  const layerNodes = [];
  const numNodes = activation.length;
  const displayNodes = Math.min(numNodes, maxNodes);
  
  for (let nodeIndex = 0; nodeIndex < displayNodes; nodeIndex++) {
    const originalIndex = numNodes <= maxNodes 
      ? nodeIndex 
      : Math.floor((nodeIndex * numNodes) / displayNodes);
    const nodeValue = activation[originalIndex];
    const nodeY = pad + (displayNodes === 1
      ? drawAreaHeight / 2
      : (drawAreaHeight * nodeIndex) / (displayNodes - 1));

    layerNodes.push({ x: layerX, y: nodeY, value: nodeValue });
  }

  return layerNodes;
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
  
  const hasModel = validateNetworkForVisualization(network);

  if (!hasModel) {
    drawEmptyNetworkMessage(ctx, canvas, containerWidth, containerHeight);
    return;
  }

  const constants = getDrawingConstants();
  const numVisualLayers = network.lastActivations.length;
  const dimensions = calculateCanvasDimensions(numVisualLayers, containerWidth, containerHeight, constants);

  canvas.width = dimensions.canvasDrawWidth;
  canvas.height = dimensions.canvasDrawHeight;

  // Draw layers and nodes
  const layerPositions = [];
  network.lastActivations.forEach((activation, layerIndex) => {
    const layerNodes = processLayerActivations(
      activation, 
      layerIndex, 
      constants.maxNodes, 
      dimensions.layerXPositions[layerIndex], 
      constants.pad, 
      dimensions.drawAreaHeight, 
      network.debug
    );

    // Draw nodes
    layerNodes.forEach(node => {
      drawNode(ctx, node.x, node.y, node.value, constants);
    });

    // Draw connections to previous layer
    if (layerPositions.length > 0) {
      drawConnections(ctx, layerPositions[layerPositions.length - 1], layerNodes, constants);
    }

    layerPositions.push(layerNodes);

    // Draw layer label
    const layerLabel = `L${layerIndex}`;
    const labelY = constants.pad;
    drawLayerLabel(ctx, dimensions.layerXPositions[layerIndex], labelY, layerLabel, constants);
  });
}