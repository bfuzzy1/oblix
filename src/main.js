import { Oblix } from './network.js';
import { OptimizedOblix } from './optimized/network.js';

import { getUIElements, addPerformanceToggle, updateDataParamUI, validateUIElements } from './ui-manager.js';
import { drawLossGraph, drawNetwork } from './graph-utils.js';
// Data utilities are imported but not yet used in the current refactoring phase
// import { generateRandomData, parseCSV, formatGeneratedDataToCSV, validateTrainingData } from './data-utils.js';

/**
 * Initializes the neural network with the specified optimization setting.
 * Business rule: Network initialization should be configurable and consistent.
 *
 * @param {boolean} useOptimized - Whether to use optimized implementation
 * @returns {Object} Neural network instance
 */
function initializeNeuralNetwork(useOptimized) {
  return useOptimized ? new OptimizedOblix(true) : new Oblix(true);
}

/**
 * Resets optimizer state arrays for the neural network.
 * Business rule: Optimizer state must be properly initialized for training.
 *
 * @param {Object} network - Neural network instance
 */
function resetOptimizerState(network) {
  network.m_dw = [];
  network.v_dw = [];
  network.m_db = [];
  network.v_db = [];
  network.m_dgamma = [];
  network.v_dgamma = [];
  network.m_dbeta = [];
  network.v_dbeta = [];
  network.s_dw = [];
  network.s_db = [];
  network.s_dgamma = [];
  network.s_dbeta = [];
}

/**
 * Restores layers to the neural network.
 * Business rule: Layer restoration maintains network architecture.
 *
 * @param {Object} network - Neural network instance
 * @param {Array} layers - Array of layer configurations
 */
function restoreLayers(network, layers) {
  if (layers.length > 0) {
    layers.forEach(layer => {
      network.addLayer(layer);
    });
  }
}

/**
 * Restores details to the neural network.
 * Business rule: Network details preserve configuration state.
 *
 * @param {Object} network - Neural network instance
 * @param {Object} details - Network details object
 */
function restoreDetails(network, details) {
  if (Object.keys(details).length > 0) {
    network.details = details;
  }
}

/**
 * Handles performance toggle changes.
 * Business rule: Performance mode switching should be seamless.
 *
 * @param {boolean} wasOptimized - Previous optimization state
 * @param {boolean} useOptimized - New optimization state
 * @param {Object} network - Current neural network instance
 * @param {Function} drawNetwork - Function to redraw network
 * @returns {Object} New neural network instance
 */
function handlePerformanceToggle(wasOptimized, useOptimized, network, drawNetwork) {
  if (wasOptimized !== useOptimized) {
    const currentLayers = network.layers ? [...network.layers] : [];
    const currentDetails = network.details ? { ...network.details } : {};
    
    const newNetwork = initializeNeuralNetwork(useOptimized);
    resetOptimizerState(newNetwork);
    restoreLayers(newNetwork, currentLayers);
    restoreDetails(newNetwork, currentDetails);
    
    drawNetwork();
    return newNetwork;
  }
  return network;
}

/**
 * Initializes the main application.
 * Business rule: Application initialization should be modular and testable.
 */
function initializeApplication() {
  // Use optimized version by default, with fallback to original
  const nn = new OptimizedOblix(true);
  const useOptimized = true;
  const lossHistory = [];

  // Get UI elements using the extracted function
  const uiElements = getUIElements();
  
  // Validate that all required UI elements exist
  if (!validateUIElements(uiElements)) {
    console.error('Required UI elements missing. Check HTML structure.');
    return;
  }
  
  // Extract commonly used elements for convenience
  const {
    lossCanvas, networkCanvas, lossCtx, networkCtx, statsEl,
    trainButton, pauseButton, resumeButton, predictButton,
    saveButton, loadButton, unloadButton, epochBar, predictionResultEl,
    numHiddenLayersInput, hiddenLayersConfigContainer, optimizerSelect,
    usePositionalEncodingCheckbox, lossFunctionSelect, l2LambdaInput,
    decayRateGroup, decayRateInput, gradientClipValueInput,
    trainingDataTextarea, testDataTextarea, epochsInput, learningRateInput,
    batchSizeInput, lrSchedulerSelect, lrStepParamsDiv, lrExpParamsDiv,
    architectureTemplateSelect, dataPatternSelect, numInputDimsInput,
    numOutputDimsInput
  } = uiElements;
  
  // Get label elements
  const numInputDimsGroup = numInputDimsInput?.closest('.input-group');
  const numOutputDimsGroup = numOutputDimsInput?.closest('.input-group');
  const inputDimsLabel = numInputDimsGroup?.querySelector('label');
  const outputDimsLabel = numOutputDimsGroup?.querySelector('label');

  // Initialize performance toggle
  addPerformanceToggle(useOptimized, (newUseOptimized) => {
    const wasOptimized = useOptimized;
    useOptimized = newUseOptimized;
    nn = handlePerformanceToggle(wasOptimized, useOptimized, nn, drawNetwork);
  });

  // Initialize event handlers
  initializeEventHandlers();

  // Initialize data utilities
  initializeDataUtilities();

  // Initialize UI creation utilities
  initializeUICreationUtilities();

  return { nn, useOptimized, lossHistory, uiElements };
}

document.addEventListener('DOMContentLoaded', () => {
  const { nn, useOptimized, lossHistory, uiElements } = initializeApplication();
  
  // Extract commonly used elements for convenience
  const {
    lossCanvas, networkCanvas, lossCtx, networkCtx, statsEl,
    trainButton, pauseButton, resumeButton, predictButton,
    saveButton, loadButton, unloadButton, epochBar, predictionResultEl,
    numHiddenLayersInput, hiddenLayersConfigContainer, optimizerSelect,
    usePositionalEncodingCheckbox, lossFunctionSelect, l2LambdaInput,
    decayRateGroup, decayRateInput, gradientClipValueInput,
    trainingDataTextarea, testDataTextarea, epochsInput, learningRateInput,
    batchSizeInput, lrSchedulerSelect, lrStepParamsDiv, lrExpParamsDiv,
    architectureTemplateSelect, dataPatternSelect, numInputDimsInput,
    numOutputDimsInput
  } = uiElements;
  
  // Get label elements
  const numInputDimsGroup = numInputDimsInput?.closest('.input-group');
  const numOutputDimsGroup = numOutputDimsInput?.closest('.input-group');
  const inputDimsLabel = numInputDimsGroup?.querySelector('label');
  const outputDimsLabel = numOutputDimsGroup?.querySelector('label');

  // Initialize performance toggle
  addPerformanceToggle(useOptimized, (newUseOptimized) => {
    const wasOptimized = useOptimized;
    useOptimized = newUseOptimized;
    nn = handlePerformanceToggle(wasOptimized, useOptimized, nn, drawNetwork);
  });

  // Initialize event handlers
  initializeEventHandlers();

  // Initialize data utilities
  initializeDataUtilities();

  // Initialize UI creation utilities
  initializeUICreationUtilities();

  function initializeDataUtilities() {
    // Data generation functions
    window.generateRandomData = generateRandomData;
    window.formatGeneratedDataToCSV = formatGeneratedDataToCSV;
    window.parseCSV = parseCSV;
  }

  function generateRandomData(
    numSamples,
    numInputs,
    numOutputs = 1,
    noiseLevel = 0.05
  ) {
    const data = [];
    for (let i = 0; i < numSamples; i++) {
      const input = [];
      for (let j = 0; j < numInputs; j++) {
        input.push(Math.random() * 2 - 1);
      }
      const output = [];
      for (let j = 0; j < numOutputs; j++) {
        const baseValue = Math.sin(input[0] * Math.PI) + Math.cos(input[1] || 0);
        const noise = (Math.random() - 0.5) * noiseLevel;
        output.push(baseValue + noise);
      }
      data.push({ input, output });
    }
    return data;
  }

  function formatGeneratedDataToCSV(dataArray) {
    if (!dataArray || dataArray.length === 0) return '';
    
    const firstSample = dataArray[0];
    const inputCount = firstSample.input.length;
    const outputCount = firstSample.output.length;
    
    let csv = '';
    
    // Header row
    for (let i = 0; i < inputCount; i++) {
      csv += `input_${i + 1},`;
    }
    for (let i = 0; i < outputCount; i++) {
      csv += `output_${i + 1}${i === outputCount - 1 ? '' : ','}`;
    }
    csv += '\n';
    
    // Data rows
    for (const sample of dataArray) {
      for (const inputValue of sample.input) {
        csv += `${inputValue},`;
      }
      for (let i = 0; i < sample.output.length; i++) {
        csv += `${sample.output[i]}${i === sample.output.length - 1 ? '' : ','}`;
      }
      csv += '\n';
    }
    
    return csv;
  }

  function parseCSV(csvString, numOutputs = 1) {
    if (!csvString || csvString.trim() === '') return [];
    
    const lines = csvString.trim().split('\n');
    if (lines.length < 2) return [];
    
    const data = [];
    const headerLine = lines[0];
    const dataLines = lines.slice(1);
    
    // Parse header to determine input/output columns
    const headers = headerLine.split(',').map(h => h.trim());
    const inputIndices = [];
    const outputIndices = [];
    
    headers.forEach((header, index) => {
      if (header.toLowerCase().startsWith('input')) {
        inputIndices.push(index);
      } else if (header.toLowerCase().startsWith('output')) {
        outputIndices.push(index);
      }
    });
    
    // If no explicit input/output headers, assume first columns are input, last are output
    if (inputIndices.length === 0 && outputIndices.length === 0) {
      const totalColumns = headers.length;
      const assumedInputCount = totalColumns - numOutputs;
      
      for (let i = 0; i < assumedInputCount; i++) {
        inputIndices.push(i);
      }
      for (let i = assumedInputCount; i < totalColumns; i++) {
        outputIndices.push(i);
      }
    }
    
    // Parse data lines
    for (const line of dataLines) {
      if (!line.trim()) continue;
      
      const values = line.split(',').map(v => v.trim());
      const input = inputIndices.map(index => parseFloat(values[index]));
      const output = outputIndices.map(index => parseFloat(values[index]));
      
      // Validate data
      if (input.some(isNaN) || output.some(isNaN)) {
        console.warn('Skipping invalid data line:', line);
        continue;
      }
      
      data.push({ input, output });
    }
    
    return data;
  }

  function initializeEventHandlers() {
    // Data pattern change handler
    dataPatternSelect.addEventListener('change', () => {
      updateDataParamUILocal(dataPatternSelect.value);
    });

    // Architecture template change handler
    architectureTemplateSelect.addEventListener('change', () => {
      const selectedTemplate = architectureTemplateSelect.value;
      if (selectedTemplate === 'custom') {
        numHiddenLayersInput.disabled = false;
        hiddenLayersConfigContainer.style.display = 'block';
      } else {
        numHiddenLayersInput.disabled = true;
        hiddenLayersConfigContainer.style.display = 'none';
        
        // Apply template
        const templateConfig = getTemplateConfig(selectedTemplate);
        if (templateConfig) {
          numHiddenLayersInput.value = templateConfig.layers;
          createLayerConfigUI(templateConfig.layers);
          applyTemplateConfig(templateConfig);
        }
      }
    });

    // Number of hidden layers change handler
    numHiddenLayersInput.addEventListener('change', () => {
      const numLayers = parseInt(numHiddenLayersInput.value) || 0;
      createLayerConfigUI(numLayers);
    });

    // Optimizer change handler
    optimizerSelect.addEventListener('change', () => {
      const selectedOptimizer = optimizerSelect.value;
      decayRateGroup.style.display = selectedOptimizer === 'adam' ? 'block' : 'none';
    });

    // Learning rate scheduler change handler
    lrSchedulerSelect.addEventListener('change', () => {
      const selectedScheduler = lrSchedulerSelect.value;
      lrStepParamsDiv.style.display = selectedScheduler === 'step' ? 'grid' : 'none';
      lrExpParamsDiv.style.display = selectedScheduler === 'exponential' ? 'grid' : 'none';
    });

    // Train button handler
    trainButton.addEventListener('click', handleTrainClick);

    // Pause/Resume button handlers
    pauseButton.addEventListener('click', handlePauseClick);
    resumeButton.addEventListener('click', handleResumeClick);

    // Predict button handler
    predictButton.addEventListener('click', handlePredictClick);

    // Save/Load button handlers
    saveButton.addEventListener('click', handleSaveClick);
    loadButton.addEventListener('click', handleLoadClick);

    // Unload button handler
    unloadButton.addEventListener('click', handleUnloadClick);

    // Window resize handler
    window.addEventListener('resize', resizeCanvases);
    setTimeout(resizeCanvases, 150);
  }

  function handleTrainClick() {
    statsEl.innerHTML = 'Starting training...';
    trainButton.disabled = true;
    trainButton.textContent = 'Training...';
    pauseButton.disabled = false;
    resumeButton.disabled = true;
    predictButton.disabled = true;
    saveButton.disabled = true;
    loadButton.disabled = true;
    unloadButton.disabled = true;
    epochBar.style.width = '0%';
    lossHistory = [];

    drawLossGraph();
    
    // Training logic will be extracted to a separate function
    executeTraining();
  }

  function handlePauseClick() {
    nn.pauseTraining();
    pauseButton.disabled = true;
    resumeButton.disabled = false;
    statsEl.innerHTML = 'Training paused';
  }

  function handleResumeClick() {
    nn.resumeTraining();
    pauseButton.disabled = false;
    resumeButton.disabled = true;
  }

  function handlePredictClick() {
    predictionResultEl.innerHTML = 'Predicting...';
    try {
      const inputStr = document.getElementById('predictionInput').value;
      if (!inputStr) throw new Error('Input empty.');
      const input = inputStr.split(',').map((s) => parseFloat(s.trim()));
      if (input.some(isNaN)) throw new Error('Invalid input.');
      if (!nn.layers || nn.layers.length === 0)
        throw new Error('Network not init.');
      const expectSz = nn.layers[0]?.inputSize;
      if (expectSz === undefined) throw new Error('Cannot get input size.');
      if (input.length !== expectSz)
        throw new Error(
          `Input size mismatch: Exp ${expectSz}, got ${input.length}.`
        );
      const pred = nn.predict(input);
      if (pred === null) throw new Error('Prediction failed.');
      const predStr = pred.map((p) => p.toFixed(5)).join(', ');
      predictionResultEl.innerHTML = `Result: [${predStr}]`;
      drawNetwork();
    } catch (error) {
      console.error('Predict error:', error);
      predictionResultEl.innerHTML = `<span class="error">Error: ${error.message}</span>`;
    }
  }

  function handleSaveClick() {
    if (!nn.layers || nn.layers.length === 0) {
      statsEl.innerHTML = '<span class="error">No model to save.</span>';
      return;
    }

    const userProvidedName = window.prompt(
      'Enter filename base (e.g., my_model):',
      'oblix_model'
    );

    if (userProvidedName === null) {
      statsEl.innerHTML = 'Save cancelled.';
      return;
    }

    let filenameBase;
    if (userProvidedName.trim() === '') {
      const now = new Date();
      const year = now.getUTCFullYear();
      const month = (now.getUTCMonth() + 1).toString().padStart(2, '0');
      const day = now.getUTCDate().toString().padStart(2, '0');
      const hours = now.getUTCHours().toString().padStart(2, '0');
      const minutes = now.getUTCMinutes().toString().padStart(2, '0');
      const seconds = now.getUTCSeconds().toString().padStart(2, '0');
      filenameBase = `oblix_${year}${month}${day}_${hours}${minutes}${seconds}Z`;
      statsEl.innerHTML = `Using default filename: ${filenameBase}.json`;
    } else {
      filenameBase = userProvidedName.trim();
    }

    nn.save(filenameBase);
    statsEl.innerHTML = `Model saved as ${filenameBase}.json`;
  }

  function handleLoadClick() {
    statsEl.innerHTML = 'Loading...';
    nn.load((error) => {
      if (error) {
        statsEl.innerHTML = `<span class="error">Load failed: ${error.message}</span>`;
        return;
      }
      const params = nn.getTotalParameters();
      statsEl.innerHTML = `<strong>Model loaded!</strong> Params: ${params.toLocaleString()}`;
      try {
        const d = nn.details?.training;
        const l = nn.layers || [];
        usePositionalEncodingCheckbox.checked =
        nn.usePositionalEncoding || false;
        if (d) {
          epochsInput.value = d.epochs || 50;
          learningRateInput.value = d.learningRate || 0.01;
          batchSizeInput.value = d.batchSize || 8;
          optimizerSelect.value = d.optimizer || 'adam';
          lossFunctionSelect.value = d.lossFunction || 'mse';
          l2LambdaInput.value = d.l2Lambda || 0;
          decayRateInput.value = d.decayRate || 0.9;
          gradientClipValueInput.value = d.gradientClipValue || 0;
          optimizerSelect.dispatchEvent(new Event('change'));
        }
        const numHid = Math.max(0, l.length - 1);
        numHiddenLayersInput.value = numHid;
        createLayerConfigUI(numHid);
        l.slice(0, numHid).forEach((layer, i) => {
          const setV = (s, v) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e && v !== undefined) e.value = v;
          };
          const setC = (s, c) => {
            const e = hiddenLayersConfigContainer.querySelector(s);
            if (e && c !== undefined) e.checked = c;
          };
          setV(
            `select[data-layer-index="${i}"][data-config-type="type"]`,
            layer.type || 'dense'
          );
          const ts = hiddenLayersConfigContainer.querySelector(
            `select[data-layer-index="${i}"][data-config-type="type"]`
          );
          if (ts) ts.dispatchEvent(new Event('change', { bubbles: true }));
          switch (layer.type) {
          case 'dense':
            setV(
              `input[data-layer-index="${i}"][data-config-type="size"]`,
              layer.outputSize
            );
            setV(
              `select[data-layer-index="${i}"][data-config-type="activation"]`,
              layer.activation
            );
            setC(
              `input[data-layer-index="${i}"][data-config-type="useBias"]`,
              layer.useBias
            );
            break;
          case 'attention':
            setV(
              `input[data-layer-index="${i}"][data-config-type="numHeads"]`,
              layer.numHeads
            );
            break;
          case 'dropout':
            setV(
              `input[data-layer-index="${i}"][data-config-type="rate"]`,
              layer.rate
            );
            break;
          }
        });

        architectureTemplateSelect.value = 'custom';
        lossHistory = [];

        drawLossGraph();
        predictionResultEl.innerHTML = 'Result: -';
        try {
          const sample =
          parseCSV(trainingDataTextarea.value, parseInt(numOutputDimsInput.value) || 1)[0];
          if (sample) nn.predict(sample.input);
        } catch (e) {}
        drawNetwork();
      } catch (uiError) {
        console.error('UI update err after load:', uiError);
        statsEl.innerHTML += ' <span class="error">(UI update failed)</span>';
      }
    });
  }

  function handleUnloadClick() {
    console.log('Unload clicked.');
    try {
      nn.reset();
      lossHistory = [];

      drawLossGraph();
      drawNetwork();
      epochBar.style.width = '0%';
      const performanceIndicator = useOptimized ? 
        '<span style="color: #4CAF50; font-weight: bold;">🚀 Optimized</span>' : 
        '<span style="color: #FF9800; font-weight: bold;">📊 Original</span>';
      statsEl.innerHTML = `Status: Model unloaded. ${performanceIndicator}`;
      predictionResultEl.innerHTML = 'Result: -';
      const defaultLayers = 2;
      numHiddenLayersInput.value = defaultLayers;
      createLayerConfigUI(defaultLayers);

      architectureTemplateSelect.value = 'custom';
      console.log('Model & UI reset.');
    } catch (error) {
      console.error('Unload error:', error);
      statsEl.innerHTML = `<span class="error">Unload error: ${error.message}</span>`;
    }
  }

  /**
   * Validates training data and extracts dimensions.
   * Business rule: Data validation prevents runtime errors.
   *
   * @param {string} trainingDataText - Training data text
   * @param {string} testDataText - Test data text
   * @param {number} outDims - Number of output dimensions
   * @returns {Object} Validated training and test data
   * @throws {Error} If data is invalid
   */
  function validateTrainingData(trainingDataText, testDataText, outDims) {
    const trainingData = parseCSV(trainingDataText, outDims);
    const testData = parseCSV(testDataText, outDims);
    
    if (trainingData.length === 0) {
      throw new Error('Training data empty/invalid.');
    }
    if (!trainingData[0]?.input || !trainingData[0]?.output) {
      throw new Error('Cannot get input/output size from data.');
    }
    
    return { trainingData, testData };
  }

  /**
   * Gets layer configuration value from UI.
   * Business rule: UI configuration must be properly extracted.
   *
   * @param {number} layerIndex - Layer index
   * @param {string} configType - Configuration type
   * @param {string} selectorType - Selector type (input/select)
   * @returns {string} Configuration value
   */
  function getLayerConfigValue(layerIndex, configType, selectorType = 'input') {
    const selector = `${selectorType}[data-layer-index="${layerIndex}"][data-config-type="${configType}"]`;
    const element = hiddenLayersConfigContainer.querySelector(selector);
    return element?.value || '';
  }

  /**
   * Gets layer configuration checkbox state from UI.
   * Business rule: UI configuration must be properly extracted.
   *
   * @param {number} layerIndex - Layer index
   * @param {string} configType - Configuration type
   * @returns {boolean} Checkbox state
   */
  function getLayerConfigCheckbox(layerIndex, configType) {
    const selector = `input[data-layer-index="${layerIndex}"][data-config-type="${configType}"]`;
    const element = hiddenLayersConfigContainer.querySelector(selector);
    return element?.checked ?? true;
  }

  /**
   * Creates dense layer configuration.
   * Business rule: Dense layers require proper size and activation settings.
   *
   * @param {number} layerIndex - Layer index
   * @param {number} inputSize - Input size
   * @param {string} weightInitMethod - Weight initialization method
   * @returns {Object} Dense layer configuration
   * @throws {Error} If configuration is invalid
   */
  function createDenseLayerConfig(layerIndex, inputSize, weightInitMethod) {
    const outputSize = parseInt(
      getLayerConfigValue(layerIndex, 'size') || '1'
    );
    if (outputSize <= 0) {
      throw new Error(`L${layerIndex + 1} Dense: Invalid nodes.`);
    }

    return {
      type: 'dense',
      inputSize,
      outputSize,
      weightInit: weightInitMethod,
      activation: getLayerConfigValue(layerIndex, 'activation', 'select') || 'tanh',
      useBias: getLayerConfigCheckbox(layerIndex, 'useBias')
    };
  }

  /**
   * Creates attention layer configuration.
   * Business rule: Attention layers require proper head count and input divisibility.
   *
   * @param {number} layerIndex - Layer index
   * @param {number} inputSize - Input size
   * @param {string} weightInitMethod - Weight initialization method
   * @returns {Object} Attention layer configuration
   * @throws {Error} If configuration is invalid
   */
  function createAttentionLayerConfig(layerIndex, inputSize, weightInitMethod) {
    const numHeads = parseInt(
      getLayerConfigValue(layerIndex, 'numHeads') || '1'
    );
    if (numHeads <= 0) {
      throw new Error(`L${layerIndex + 1} Attn: Invalid heads.`);
    }
    if (inputSize % numHeads !== 0) {
      throw new Error(
        `L${layerIndex + 1} Attn: Input ${inputSize} not divisible by ${numHeads} heads.`
      );
    }

    return {
      type: 'attention',
      inputSize,
      outputSize: inputSize,
      weightInit: weightInitMethod,
      numHeads
    };
  }

  /**
   * Creates dropout layer configuration.
   * Business rule: Dropout rate must be between 0 and 1.
   *
   * @param {number} layerIndex - Layer index
   * @param {number} inputSize - Input size
   * @param {string} weightInitMethod - Weight initialization method
   * @returns {Object} Dropout layer configuration
   * @throws {Error} If configuration is invalid
   */
  function createDropoutLayerConfig(layerIndex, inputSize, weightInitMethod) {
    const rate = parseFloat(
      getLayerConfigValue(layerIndex, 'rate') || '0'
    );
    if (rate < 0 || rate >= 1) {
      throw new Error(`L${layerIndex + 1} Dropout: Invalid rate.`);
    }

    return {
      type: 'dropout',
      inputSize,
      outputSize: inputSize,
      weightInit: weightInitMethod,
      rate
    };
  }

  /**
   * Creates simple layer configuration (layernorm, softmax).
   * Business rule: Simple layers maintain input size as output size.
   *
   * @param {number} layerIndex - Layer index
   * @param {string} layerType - Layer type
   * @param {number} inputSize - Input size
   * @param {string} weightInitMethod - Weight initialization method
   * @returns {Object} Layer configuration
   */
  function createSimpleLayerConfig(layerIndex, layerType, inputSize, weightInitMethod) {
    return {
      type: layerType,
      inputSize,
      outputSize: inputSize,
      weightInit: weightInitMethod
    };
  }

  /**
   * Creates layer configuration based on type.
   * Business rule: Each layer type has specific configuration requirements.
   *
   * @param {number} layerIndex - Layer index
   * @param {string} layerType - Layer type
   * @param {number} inputSize - Input size
   * @param {string} weightInitMethod - Weight initialization method
   * @returns {Object} Layer configuration
   * @throws {Error} If layer type is unknown
   */
  function createLayerConfig(layerIndex, layerType, inputSize, weightInitMethod) {
    switch (layerType) {
      case 'dense':
        return createDenseLayerConfig(layerIndex, inputSize, weightInitMethod);
      case 'attention':
        return createAttentionLayerConfig(layerIndex, inputSize, weightInitMethod);
      case 'dropout':
        return createDropoutLayerConfig(layerIndex, inputSize, weightInitMethod);
      case 'layernorm':
      case 'softmax':
        return createSimpleLayerConfig(layerIndex, layerType, inputSize, weightInitMethod);
      default:
        throw new Error(`L${layerIndex + 1}: Unknown type "${layerType}".`);
    }
  }

  /**
   * Builds layer configurations from UI settings.
   * Business rule: Layer configurations must be properly validated and ordered.
   *
   * @param {number} numHidden - Number of hidden layers
   * @param {number} numInputs - Number of inputs
   * @param {string} weightInitMethod - Weight initialization method
   * @returns {Array} Layer configurations
   * @throws {Error} If configuration is invalid
   */
  function buildLayerConfigurations(numHidden, numInputs, weightInitMethod) {
    const layerCfgs = [];
    let currentInputSize = numInputs;

    for (let i = 0; i < numHidden; i++) {
      const layerType = getLayerConfigValue(i, 'type', 'select') || 'dense';
      const cfg = createLayerConfig(i, layerType, currentInputSize, weightInitMethod);
      
      if (!cfg.outputSize) {
        throw new Error(`L${i + 1}: No output size.`);
      }
      
      layerCfgs.push(cfg);
      currentInputSize = cfg.outputSize;
    }

    return layerCfgs;
  }

  /**
   * Determines final activation based on loss function and architecture.
   * Business rule: Final activation must match loss function requirements.
   *
   * @param {string} selectedLoss - Selected loss function
   * @param {boolean} isAutoencoder - Whether this is an autoencoder
   * @param {number} numOutputs - Number of outputs
   * @returns {string} Final activation function
   */
  function determineFinalActivation(selectedLoss, isAutoencoder, numOutputs) {
    if (selectedLoss === 'crossentropy') {
      if (isAutoencoder) {
        const finalAct = architectureTemplates['autoencoder']?.finalActivationHint || 'sigmoid';
        console.log(
          `Autoencoder with Cross-Entropy? Using template hint or sigmoid: ${finalAct}`
        );
        return finalAct;
      } else if (numOutputs > 1) {
        return 'softmax';
      } else if (numOutputs === 1) {
        return 'sigmoid';
      }
    }
    return 'none';
  }

  /**
   * Executes the neural network training process.
   * Business rule: Training must be properly configured and validated.
   */
  async function executeTraining() {
    try {
      const weightInitMethod = document.getElementById('weightInit').value;
      const outDims = parseInt(numOutputDimsInput.value) || 1;
      
      // Validate training data
      const { trainingData, testData } = validateTrainingData(
        trainingDataTextarea.value, 
        testDataTextarea.value, 
        outDims
      );

      // Reset network
      nn.reset();

      // Build layer configurations
      const numHidden = parseInt(numHiddenLayersInput.value);
      const numInputs = trainingData[0].input.length;
      const layerCfgs = buildLayerConfigurations(numHidden, numInputs, weightInitMethod);

      // Determine output configuration
      const isAutoencoder = architectureTemplateSelect.value === 'autoencoder';
      const numOutputs = isAutoencoder ? numInputs : trainingData[0].output.length;
      if (numOutputs <= 0) {
        throw new Error('Zero output cols defined.');
      }

      const selectedLoss = lossFunctionSelect.value;
      const finalActivation = determineFinalActivation(selectedLoss, isAutoencoder, numOutputs);

      // Add output layer
      layerCfgs.push({
        type: 'dense',
        inputSize: layerCfgs.length > 0 ? layerCfgs[layerCfgs.length - 1].outputSize : numInputs,
        outputSize: numOutputs,
        weightInit: weightInitMethod,
        activation: finalActivation,
        useBias: true
      });

      // Build network
      layerCfgs.forEach(cfg => nn.layer(cfg));

      // Get training parameters
      const epochs = parseInt(epochsInput.value) || 100;
      const learningRate = parseFloat(learningRateInput.value) || 0.01;
      const batchSize = parseInt(batchSizeInput.value) || 1;
      const optimizer = optimizerSelect.value || 'adam';
      const l2Lambda = parseFloat(l2LambdaInput.value) || 0;
      const gradientClipValue = parseFloat(gradientClipValueInput.value) || 0;
      const decayRate = parseFloat(decayRateInput.value) || 0.9;

      // Start training
      const trainingResult = await nn.train(
        trainingData,
        testData,
        {
          epochs,
          learningRate,
          batchSize,
          optimizer,
          l2Lambda,
          gradientClipValue,
          decayRate,
          lossFunction: selectedLoss,
          usePositionalEncoding: usePositionalEncodingCheckbox.checked,
          lrSchedule: lrSchedulerSelect.value,
          lrStepDecayFactor: parseFloat(document.getElementById('lrStepDecayFactor')?.value) || 0.5,
          lrStepDecaySize: parseInt(document.getElementById('lrStepDecaySize')?.value) || 10,
          lrExpDecayRate: parseFloat(document.getElementById('lrExpDecayRate')?.value) || 0.95
        },
        {
          onEpoch: (epoch, loss, testLoss, accuracy, rSquared) => {
            // Update progress bar
            const progress = (epoch / epochs) * 100;
            epochBar.style.width = `${progress}%`;
            epochBar.textContent = `${epoch}/${epochs}`;

            // Update loss history
            lossHistory.push(loss);

            // Update UI
            statsEl.textContent = `Epoch ${epoch}: Loss=${loss.toFixed(4)}, Test=${testLoss?.toFixed(4) || 'N/A'}, Acc=${accuracy?.toFixed(3) || 'N/A'}, R²=${rSquared?.toFixed(3) || 'N/A'}`;

                         // Draw visualizations
             drawLossGraphLocal();
             drawNetwork();
          }
        }
      );

      console.log('Training completed:', trainingResult);
      
      // Update UI after training
      statsEl.textContent = `Training complete! Final loss: ${trainingResult.finalLoss.toFixed(4)}`;
      
    } catch (error) {
      console.error('Training error:', error);
      statsEl.textContent = `Training error: ${error.message}`;
    }
  }

  function getTemplateConfig(template) {
    const templates = {
      'simple': { layers: 1, config: [{ type: 'dense', units: 10, activation: 'relu' }] },
      'medium': { layers: 2, config: [
        { type: 'dense', units: 20, activation: 'relu' },
        { type: 'dense', units: 10, activation: 'relu' }
      ]},
      'complex': { layers: 3, config: [
        { type: 'dense', units: 50, activation: 'relu' },
        { type: 'dense', units: 25, activation: 'relu' },
        { type: 'dense', units: 10, activation: 'relu' }
      ]}
    };
    return templates[template];
  }

  function applyTemplateConfig(config) {
    // Apply template configuration to UI
    if (config && config.config) {
      config.config.forEach((layerConfig, index) => {
        const typeSelect = hiddenLayersConfigContainer.querySelector(
          `select[data-layer-index="${index}"][data-config-type="type"]`
        );
        const unitsInput = hiddenLayersConfigContainer.querySelector(
          `input[data-layer-index="${index}"][data-config-type="units"]`
        );
        const activationSelect = hiddenLayersConfigContainer.querySelector(
          `select[data-layer-index="${index}"][data-config-type="activation"]`
        );

        if (typeSelect) typeSelect.value = layerConfig.type;
        if (unitsInput) unitsInput.value = layerConfig.units;
        if (activationSelect) activationSelect.value = layerConfig.activation;
      });
    }
  }

  function updateDataParamUILocal(selectedPattern) {
    // Use the extracted function from ui-manager.js
    updateDataParamUI(selectedPattern, uiElements);
    
    // Additional UI logic specific to main.js
    let ignoreDims = false;
    let inputTitle = 'Number of input features per sample';
    let outputTitle = 'Number of output values per sample';

    switch (selectedPattern) {
    case 'xor':
    case 'circular':
      ignoreDims = true;
      inputTitle = 'Input Dimensions (Fixed to 2 for this pattern)';
      outputTitle = 'Output Dimensions (Fixed to 1 for this pattern)';
      break;
    case 'blobs':
      ignoreDims = true;
      inputTitle = 'Input Dimensions (Fixed to 2 for blobs)';
      outputTitle = 'Number of Classes (Output Dim controls this)';
      break;
    case 'linear':
    case 'random':
    default:
      ignoreDims = false;
      break;
    }

    if (numInputDimsInput) numInputDimsInput.disabled = ignoreDims;
    if (numOutputDimsInput) numOutputDimsInput.disabled = ignoreDims;

    if (numInputDimsGroup)
      numInputDimsGroup.classList.toggle('ignored-param', ignoreDims);
    if (numOutputDimsGroup)
      numOutputDimsGroup.classList.toggle('ignored-param', ignoreDims);

    if (inputDimsLabel) inputDimsLabel.title = inputTitle;
    if (outputDimsLabel) outputDimsLabel.title = outputTitle;
  }

  if (dataPatternSelect) {
    dataPatternSelect.addEventListener('change', (event) => {
      updateDataParamUILocal(event.target.value);
    });

    updateDataParamUILocal(dataPatternSelect.value);
  } else {
    console.error('Data Pattern select element not found.');
  }

  const architectureTemplates = {
    mlp: {
      numHidden: 2,
      layers: [
        { type: 'dense', size: 16, activation: 'relu', useBias: true },
        { type: 'dense', size: 8, activation: 'relu', useBias: true }
      ]
    },
    autoencoder: {
      numHidden: 3,
      layers: [
        { type: 'dense', size: 16, activation: 'relu', useBias: true },
        { type: 'dense', size: 4, activation: 'relu', useBias: true },
        { type: 'dense', size: 16, activation: 'relu', useBias: true }
      ],

      finalActivationHint: 'sigmoid'
    },
    transformerEncoder: {
      numHidden: 8,
      layers: [
        { type: 'layernorm', size: null },
        { type: 'attention', numHeads: 3 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 },

        { type: 'layernorm', size: null },
        { type: 'attention', numHeads: 4 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 }
      ],

      finalActivationHint: 'none'
    },
    residualAttention: {
      numHidden: 5,
      layers: [
        { type: 'layernorm' },
        { type: 'attention', numHeads: 3 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 },
        { type: 'layernorm' }
      ],
      finalActivationHint: 'none'
    },
    mlpDropout: {
      numHidden: 4,
      layers: [
        { type: 'dense', size: 16, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.2 },
        { type: 'dense', size: 8, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.2 }
      ],
      finalActivationHint: 'none'
    },
    deepResidualMLP: {
      numHidden: 5,
      layers: [
        { type: 'layernorm' },
        { type: 'dense', size: 64, activation: 'relu', useBias: true },
        { type: 'dense', size: 64, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 },
        { type: 'layernorm' }
      ],
      finalActivationHint: 'none'
    },
    transformerStack: {
      numHidden: 16,
      layers: [
        { type: 'layernorm', size: null },
        { type: 'attention', numHeads: 3 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 },

        { type: 'layernorm', size: null },
        { type: 'attention', numHeads: 4 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 },

        { type: 'layernorm', size: null },
        { type: 'attention', numHeads: 3 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 },

        { type: 'layernorm', size: null },
        { type: 'attention', numHeads: 4 },
        { type: 'dense', size: 32, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.1 }
      ],
      finalActivationHint: 'none'
    },
    autoencoderDropout: {
      numHidden: 5,
      layers: [
        { type: 'dense', size: 16, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.2 },
        { type: 'dense', size: 4, activation: 'relu', useBias: true },
        { type: 'dropout', rate: 0.2 },
        { type: 'dense', size: 16, activation: 'relu', useBias: true }
      ],
      finalActivationHint: 'sigmoid'
    },
    softmaxClassifier: {
      numHidden: 1,
      layers: [
        { type: 'dense', size: 16, activation: 'relu', useBias: true }
      ],
      finalActivationHint: 'softmax'
    }
  };

  try {
    function _wm(c) {
      if (!c) return;
      const d = Math.max(
          0,
          Math.floor((new Date() - new Date(2023, 0, 1)) / 864e5)
        ),
        r = 0x4f,
        g = (d >> 8) & 255,
        b = d & 255;
      c.fillStyle = `rgb(${r},${g},${b})`;
      c.fillRect(0, 0, 1, 1);
    }
    _wm(document.getElementById('watermarkCanvas')?.getContext('2d'));
  } catch (e) {
    console.error('Watermark err:', e);
  }

  function createLayerConfigUI(numLayers) {
    if (!hiddenLayersConfigContainer) return;
    
    hiddenLayersConfigContainer.innerHTML = '';
    
    for (let i = 0; i < numLayers; i++) {
      const layerDiv = document.createElement('div');
      layerDiv.className = 'layer-config';
      layerDiv.innerHTML = `
        <h4>Layer ${i + 1}</h4>
        <div class="layer-controls">
          <div class="input-group">
            <label for="layer${i}Type">Type:</label>
            <select id="layer${i}Type" data-layer-index="${i}" data-config-type="type">
              <option value="dense">Dense</option>
              <option value="attention">Attention</option>
              <option value="dropout">Dropout</option>
              <option value="layernorm">LayerNorm</option>
              <option value="softmax">Softmax</option>
            </select>
          </div>
          <div class="input-group">
            <label for="layer${i}Size">Size:</label>
            <input type="number" id="layer${i}Size" data-layer-index="${i}" data-config-type="size" value="10" min="1" max="1000">
          </div>
          <div class="input-group">
            <label for="layer${i}Activation">Activation:</label>
            <select id="layer${i}Activation" data-layer-index="${i}" data-config-type="activation">
              <option value="relu">ReLU</option>
              <option value="tanh">Tanh</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="softmax">Softmax</option>
              <option value="none">None</option>
            </select>
          </div>
          <div class="input-group">
            <label for="layer${i}UseBias">Use Bias:</label>
            <input type="checkbox" id="layer${i}UseBias" data-layer-index="${i}" data-config-type="useBias" checked>
          </div>
        </div>
      `;
      
      hiddenLayersConfigContainer.appendChild(layerDiv);
      
      // Add event listener for type changes
      const typeSelect = layerDiv.querySelector(`#layer${i}Type`);
      typeSelect.addEventListener('change', () => {
        updateLayerOptionsUI(i, typeSelect.value);
      });
    }
  }

  function updateLayerOptionsUI(layerIndex, layerType) {
    const layerDiv = hiddenLayersConfigContainer.children[layerIndex];
    if (!layerDiv) return;
    
    const sizeInput = layerDiv.querySelector(`#layer${layerIndex}Size`);
    const activationSelect = layerDiv.querySelector(`#layer${layerIndex}Activation`);
    const useBiasCheckbox = layerDiv.querySelector(`#layer${layerIndex}UseBias`);
    
    // Hide all optional controls
    sizeInput.parentElement.style.display = 'none';
    activationSelect.parentElement.style.display = 'none';
    useBiasCheckbox.parentElement.style.display = 'none';
    
    // Show relevant controls based on layer type
    switch (layerType) {
      case 'dense':
        sizeInput.parentElement.style.display = 'block';
        activationSelect.parentElement.style.display = 'block';
        useBiasCheckbox.parentElement.style.display = 'block';
        break;
      case 'attention':
        // Add attention-specific controls if needed
        break;
      case 'dropout':
        // Add dropout-specific controls if needed
        break;
      case 'layernorm':
      case 'softmax':
        // These layers don't need additional controls
        break;
    }
  }

  function drawLossGraphLocal() {
    if (!lossCtx || !lossCanvas) return;
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    if (lossHistory.length < 2) return;
    
    const trainLosses = lossHistory
      .map((history) => history.train)
      .filter((loss) => loss !== null && isFinite(loss));
    const testLosses = lossHistory
      .map((history) => history.test)
      .filter((loss) => loss !== null && isFinite(loss));
    
    let maxLoss = 0.1;
    if (trainLosses.length > 0) maxLoss = Math.max(maxLoss, ...trainLosses);
    if (testLosses.length > 0) maxLoss = Math.max(maxLoss, ...testLosses);
    maxLoss = Math.max(maxLoss, 0.1);
    
    const canvasWidth = lossCanvas.width;
    const canvasHeight = lossCanvas.height;
    const numPoints = lossHistory.length;
    const plotHeight = canvasHeight * 0.9;
    const yOffset = canvasHeight * 0.05;
    
    const plot = (context, points, color) => {
      context.strokeStyle = color;
      context.lineWidth = 1.5;
      context.beginPath();
      let isFirst = true;
      points.forEach((point, index) => {
        if (point !== null && isFinite(point)) {
          const x = (index / Math.max(1, numPoints - 1)) * canvasWidth;
          const y = canvasHeight - (point / maxLoss) * plotHeight - yOffset;
          if (isFirst) {
            context.moveTo(x, y);
            isFirst = false;
          } else {
            context.lineTo(x, y);
          }
        } else {
          isFirst = true;
        }
      });
      context.stroke();
    };
    
    const trainColor = getComputedStyle(document.body).getPropertyValue('--text')?.trim() || '#fff';
    plot(
      lossCtx,
      lossHistory.map((history) => history.train),
      trainColor
    );
    plot(
      lossCtx,
      lossHistory.map((history) => history.test),
      '#87CEEB'
    );
  }

  function resizeCanvases() {
    const lossContainer = lossCanvas?.parentElement;
    const networkContainer = networkCanvas?.parentElement;
    
    if (lossContainer && lossCanvas) {
      lossCanvas.width = lossContainer.clientWidth;
      lossCanvas.height = lossContainer.clientHeight;
      drawLossGraph();
    }
    
    if (networkContainer && networkCanvas) {
      networkCanvas.width = networkContainer.clientWidth;
      networkCanvas.height = networkContainer.clientHeight;
      drawNetwork();
    }
  }
});

