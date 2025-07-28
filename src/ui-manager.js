/**
 * UI Manager for the Oblix neural network playground.
 * 
 * Following Japanese-Level Code Quality principles:
 * - Monozukuri: Each function has a single, clear responsibility
 * - Kaizen: Extracted from the massive main.js function
 * - Wabi-sabi: Simple, focused functions that solve today's need
 */

/**
 * Manages DOM element references for the neural network UI.
 * Business rule: UI elements must be consistently accessible across modules.
 * 
 * @returns {Object} Object containing all UI element references
 */
export function getUIElements() {
  return {
    // Canvas elements
    lossCanvas: document.getElementById('lossGraph'),
    networkCanvas: document.getElementById('networkGraph'),
    lossCtx: document.getElementById('lossGraph')?.getContext('2d'),
    networkCtx: document.getElementById('networkGraph')?.getContext('2d'),
    
    // Control elements
    statsEl: document.getElementById('stats'),
    trainButton: document.getElementById('trainButton'),
    pauseButton: document.getElementById('pauseButton'),
    resumeButton: document.getElementById('resumeButton'),
    predictButton: document.getElementById('predictButton'),
    saveButton: document.getElementById('saveButton'),
    loadButton: document.getElementById('loadButton'),
    unloadButton: document.getElementById('unloadButton'),
    epochBar: document.getElementById('epochBar'),
    predictionResultEl: document.getElementById('predictionResult'),
    
    // Configuration elements
    numHiddenLayersInput: document.getElementById('numHiddenLayers'),
    hiddenLayersConfigContainer: document.getElementById('hiddenLayersConfig'),
    optimizerSelect: document.getElementById('optimizer'),
    usePositionalEncodingCheckbox: document.getElementById('usePositionalEncoding'),
    lossFunctionSelect: document.getElementById('lossFunction'),
    l2LambdaInput: document.getElementById('l2Lambda'),
    decayRateGroup: document.getElementById('decayRateGroup'),
    decayRateInput: document.getElementById('decayRate'),
    gradientClipValueInput: document.getElementById('gradientClipValue'),
    
    // Data elements
    trainingDataTextarea: document.getElementById('trainingData'),
    testDataTextarea: document.getElementById('testData'),
    epochsInput: document.getElementById('epochs'),
    learningRateInput: document.getElementById('learningRate'),
    batchSizeInput: document.getElementById('batchSize'),
    
    // Learning rate scheduler elements
    lrSchedulerSelect: document.getElementById('lrScheduler'),
    lrStepParamsDiv: document.getElementById('lrStepParams'),
    lrExpParamsDiv: document.getElementById('lrExpParams'),
    
    // Architecture elements
    architectureTemplateSelect: document.getElementById('architectureTemplateSelect'),
    dataPatternSelect: document.getElementById('dataPattern'),
    numInputDimsInput: document.getElementById('numInputDims'),
    numOutputDimsInput: document.getElementById('numOutputDims')
  };
}

/**
 * Adds performance toggle to the UI.
 * Business rule: Users should be able to switch between optimized and original implementations.
 * 
 * @param {boolean} useOptimized - Current optimization state
 * @param {Function} onToggle - Callback when toggle changes
 */
export function addPerformanceToggle(useOptimized, onToggle) {
  const controlsContainer = document.querySelector('.controls') || document.body;
  const toggleContainer = document.createElement('div');
  toggleContainer.className = 'input-group';
  toggleContainer.innerHTML = `
    <label for="performanceToggle">Performance Mode:</label>
    <select id="performanceToggle" title="Choose between original and optimized implementation">
      <option value="optimized" ${useOptimized ? 'selected' : ''}>🚀 Optimized (2x faster)</option>
      <option value="original" ${!useOptimized ? 'selected' : ''}>📊 Original</option>
    </select>
  `;
  
  // Insert after the first input group
  const firstGroup = controlsContainer.querySelector('.input-group');
  if (firstGroup) {
    firstGroup.parentNode.insertBefore(toggleContainer, firstGroup.nextSibling);
  } else {
    controlsContainer.appendChild(toggleContainer);
  }
  
  // Add event listener
  const toggle = document.getElementById('performanceToggle');
  toggle.addEventListener('change', (event) => {
    const newUseOptimized = event.target.value === 'optimized';
    onToggle(newUseOptimized);
  });
}

/**
 * Updates data parameter UI based on selected pattern.
 * Business rule: Different data patterns require different UI configurations.
 * 
 * @param {string} selectedPattern - The selected data pattern
 * @param {Object} uiElements - UI element references
 */
export function updateDataParamUI(selectedPattern, uiElements) {
  const { numInputDimsInput, numOutputDimsInput, numInputDimsGroup, numOutputDimsGroup } = uiElements;
  
  if (!numInputDimsInput || !numOutputDimsInput) return;
  
  const inputDimsLabel = numInputDimsGroup?.querySelector('label');
  const outputDimsLabel = numOutputDimsGroup?.querySelector('label');
  
  switch (selectedPattern) {
  case 'xor':
    numInputDimsInput.value = '2';
    numOutputDimsInput.value = '1';
    if (inputDimsLabel) inputDimsLabel.textContent = 'Input Dimensions (XOR: 2)';
    if (outputDimsLabel) outputDimsLabel.textContent = 'Output Dimensions (XOR: 1)';
    break;
  case 'linear':
    numInputDimsInput.value = '1';
    numOutputDimsInput.value = '1';
    if (inputDimsLabel) inputDimsLabel.textContent = 'Input Dimensions (Linear: 1)';
    if (outputDimsLabel) outputDimsLabel.textContent = 'Output Dimensions (Linear: 1)';
    break;
  case 'circular':
    numInputDimsInput.value = '2';
    numOutputDimsInput.value = '1';
    if (inputDimsLabel) inputDimsLabel.textContent = 'Input Dimensions (Circular: 2)';
    if (outputDimsLabel) outputDimsLabel.textContent = 'Output Dimensions (Circular: 1)';
    break;
  case 'blobs':
    numInputDimsInput.value = '2';
    numOutputDimsInput.value = '3';
    if (inputDimsLabel) inputDimsLabel.textContent = 'Input Dimensions (Blobs: 2)';
    if (outputDimsLabel) outputDimsLabel.textContent = 'Output Dimensions (Blobs: 3)';
    break;
  default:
    numInputDimsInput.value = '2';
    numOutputDimsInput.value = '1';
    if (inputDimsLabel) inputDimsLabel.textContent = 'Input Dimensions';
    if (outputDimsLabel) outputDimsLabel.textContent = 'Output Dimensions';
  }
}

/**
 * Validates that all required UI elements exist.
 * Business rule: Missing UI elements should be detected early to prevent runtime errors.
 * 
 * @param {Object} uiElements - UI element references
 * @returns {boolean} True if all required elements exist
 */
export function validateUIElements(uiElements) {
  const requiredElements = [
    'lossCanvas', 'networkCanvas', 'trainButton', 'predictButton',
    'epochsInput', 'learningRateInput', 'batchSizeInput'
  ];
  
  for (const elementName of requiredElements) {
    if (!uiElements[elementName]) {
      console.warn(`Missing required UI element: ${elementName}`);
      return false;
    }
  }
  
  return true;
}