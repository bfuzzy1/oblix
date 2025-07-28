import { Oblix } from './network.js';
import { OptimizedOblix } from './optimized/network.js';
import { oblixUtils } from './utils.js';
import { getUIElements, addPerformanceToggle, updateDataParamUI, validateUIElements } from './ui-manager.js';
import { drawLossGraph, drawNetwork } from './graph-utils.js';
// Data utilities are imported but not yet used in the current refactoring phase
// import { generateRandomData, parseCSV, formatGeneratedDataToCSV, validateTrainingData } from './data-utils.js';

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
  // Use optimized version by default, with fallback to original
    let nn = new OptimizedOblix(true);
    let useOptimized = true;
    let lossHistory = [];
  
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
    
      if (wasOptimized !== useOptimized) {
        // Recreate network with new implementation
        const currentLayers = nn.layers ? [...nn.layers] : [];
        const currentDetails = nn.details ? { ...nn.details } : {};
      
        nn = useOptimized ? new OptimizedOblix(true) : new Oblix(true);
        // Reset optimizer state arrays
        nn.m_dw = [];
        nn.v_dw = [];
        nn.m_db = [];
        nn.v_db = [];
        nn.m_dgamma = [];
        nn.v_dgamma = [];
        nn.m_dbeta = [];
        nn.v_dbeta = [];
        nn.s_dw = [];
        nn.s_db = [];
        nn.s_dgamma = [];
        nn.s_dbeta = [];
        
        // Restore layers if any
        if (currentLayers.length > 0) {
          currentLayers.forEach(layer => {
            nn.addLayer(layer);
          });
        }
        
        // Restore details if any
        if (Object.keys(currentDetails).length > 0) {
          nn.details = currentDetails;
        }
        
        drawNetwork();
      }
    });

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

    function formatGeneratedDataToCSV(dataArray) {
      if (!Array.isArray(dataArray)) return [];
      return dataArray
        .map((sample) => {
          if (
            !sample ||
          !Array.isArray(sample.input) ||
          !Array.isArray(sample.output)
          ) {
            console.warn('Skipping invalid sample in formatGeneratedDataToCSV');
            return null;
          }

          return [...sample.input, ...sample.output]
            .map((v) =>
              typeof v === 'number' && isFinite(v) ? v.toFixed(3) : 'NaN'
            )
            .join(', ');
        })
        .filter((row) => row !== null);
    }

    function generateRandomData(
      numSamples,
      numInputs,
      numOutputs = 1,
      noiseLevel = 0.05
    ) {
      if (numInputs <= 0 || numOutputs <= 0) return '';
      const data = [];
      for (let i = 0; i < numSamples; i++) {
        const input = [];
        for (let j = 0; j < numInputs; j++) {
        // Ensure input values are between 0.01 and 0.99
          input.push(0.01 + Math.random() * 0.98);
        }
        const output = [];
        for (let j = 0; j < numOutputs; j++) {
          const base = Math.sin(input[0] * Math.PI * 2) * 0.4 + 0.5;
          const noise = (Math.random() - 0.5) * 2 * noiseLevel;
          const final = Math.max(0.01, Math.min(0.99, base + noise));
          output.push(final);
        }
        data.push([...input, ...output].map((v) => v.toFixed(3)).join(', '));
      }
      return data.join('\n');
    }
    document.getElementById('generateDataBtn').addEventListener('click', () => {
      const numTrainInput =
      parseInt(document.getElementById('numTrainSamples').value) || 100;
      const numTestInput =
      parseInt(document.getElementById('numTestSamples').value) || 25;
      const numIn = parseInt(document.getElementById('numInputDims').value) || 3;
      const numOut =
      parseInt(document.getElementById('numOutputDims').value) || 1;
      const noise =
      parseFloat(document.getElementById('noiseLevel').value) || 0.05;
      const selectedPattern = document.getElementById('dataPattern').value;

      const safeNumTrain = Math.max(1, Math.min(5000, numTrainInput));
      const safeNumTest = Math.max(0, Math.min(1000, numTestInput));
      const safeNumIn = Math.max(1, Math.min(50, numIn));
      const safeNumOut = Math.max(1, Math.min(20, numOut));
      const safeNoise = Math.max(0, Math.min(1, noise));

      const totalSamples = safeNumTrain + safeNumTest;
      let generatedData = [];
      let patternDesc = 'Default Random';
      let finalInputDims = safeNumIn;
      let finalOutputDims = safeNumOut;

      try {
        switch (selectedPattern) {
        case 'xor':
          generatedData = oblixUtils.generateXORData(totalSamples, safeNoise);
          patternDesc = 'XOR';
          finalInputDims = 2;
          finalOutputDims = 1;
          break;
        case 'linear':
          generatedData = oblixUtils.generateLinearData(
            totalSamples,
            safeNoise,
            safeNumIn,
            safeNumOut
          );
          patternDesc = 'Linear';

          break;
        case 'circular':
          generatedData = oblixUtils.generateCircularData(
            totalSamples,
            safeNoise
          );
          patternDesc = 'Circular';
          finalInputDims = 2;
          finalOutputDims = 1;
          break;
        case 'blobs':
          const numClasses = Math.max(2, safeNumOut);
          generatedData = oblixUtils.generateGaussianBlobs(
            totalSamples,
            safeNoise,
            numClasses
          );
          patternDesc = `${numClasses}-Class Gaussian Blobs`;
          finalInputDims = 2;
          finalOutputDims = 1;
          break;
        case 'random':
        default:
          const randomCsvString = generateRandomData(
            totalSamples,
            safeNumIn,
            safeNumOut,
            safeNoise
          );
          generatedData = randomCsvString.split('\n');
          patternDesc = 'Default Random';
          break;
        }

        let csvRows = [];
        if (selectedPattern === 'random') {
          csvRows = generatedData;
        } else {
          csvRows = formatGeneratedDataToCSV(generatedData);
        }

        if (csvRows.length < totalSamples) {
          console.warn(
            `Generated fewer rows (${csvRows.length}) than requested (${totalSamples}). Adjusting counts.`
          );
        }
        const actualTotal = csvRows.length;
        const actualTrainCount = Math.min(safeNumTrain, actualTotal);
        const actualTestCount = Math.min(
          safeNumTest,
          actualTotal - actualTrainCount
        );

        for (let i = actualTotal - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [csvRows[i], csvRows[j]] = [csvRows[j], csvRows[i]];
        }

        const trainCsvString = csvRows.slice(0, actualTrainCount).join('\n');
        const testCsvString = csvRows
          .slice(actualTrainCount, actualTrainCount + actualTestCount)
          .join('\n');

        trainingDataTextarea.value = trainCsvString;
        testDataTextarea.value = testCsvString;

        statsEl.innerHTML = `Generated ${actualTrainCount}/${actualTestCount} samples using <strong>${patternDesc}</strong> pattern (${finalInputDims} inputs, ${finalOutputDims} outputs, ${safeNoise.toFixed(2)} noise).`;

        try {
          const firstSample = parseCSV(trainCsvString, finalOutputDims)[0];
          if (firstSample && nn.layers && nn.layers.length > 0) {
            if (nn.layers[0]?.inputSize === firstSample.input.length) {
              nn.predict(firstSample.input);
            } else {
              console.warn(
                'Model input size doesn\'t match newly generated data. Skipping predict/redraw.'
              );
            }
          }
        } catch (e) {
          console.warn('Error during post-generation predict:', e);
        }

        drawNetwork();
      } catch (error) {
        console.error('Error during data generation:', error);
        statsEl.innerHTML = `<span class="error">Generation Error: ${error.message}</span>`;
      }
    });
    function parseCSV(csvString, numOutputs = 1) {
      if (!csvString || typeof csvString !== 'string') return [];
      const outputs = Math.max(1, Math.floor(numOutputs));
      return csvString
        .trim()
        .split('\n')
        .map((r) => r.trim())
        .filter((r) => r.length > 0)
        .map((r, idx) => {
          const vals = r.split(',').map((v) => parseFloat(v.trim()));
          if (vals.some(isNaN)) {
            console.warn(`R ${idx + 1} NaN`);
            return null;
          }
          if (vals.length < outputs + 1) {
            console.warn(`R ${idx + 1} insufficient vals`);
            return null;
          }
          const input = vals.slice(0, vals.length - outputs);
          const output = vals.slice(vals.length - outputs);
          if (input.length === 0 || output.length !== outputs) {
            console.warn(`R ${idx + 1} invalid io lens`);
            return null;
          }
          return { input: input, output: output };
        })
        .filter((i) => i !== null);
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

    function createLayerConfigUI(numLayers) {
      hiddenLayersConfigContainer.innerHTML = '';
      const activationTypes = [
        'tanh',
        'sigmoid',
        'relu',
        'leakyrelu',
        'gelu',
        'selu',
        'swish',
        'mish',
        'softmax',
        'none'
      ];
      const layerTypes = [
        'dense',
        'layernorm',
        'attention',
        'dropout',
        'softmax'
      ];
      if (numLayers === 0) {
        hiddenLayersConfigContainer.innerHTML =
        '<p class="layer-note">No hidden layers. Direct input-to-output connection (final layer added automatically).</p>';
        return;
      }
      for (let i = 0; i < numLayers; i++) {
        const layerGroup = document.createElement('div');
        layerGroup.className = 'input-group settings-grid';
        const typeDiv = document.createElement('div');
        typeDiv.className = 'input-group';
        const typeLabel = document.createElement('label');
        typeLabel.textContent = `Layer ${i + 1} Type:`;
        typeLabel.htmlFor = `layerType_${i}`;
        typeLabel.title = 'Choose the operation for this layer...';
        const typeSelect = document.createElement('select');
        typeSelect.id = `layerType_${i}`;
        typeSelect.dataset.layerIndex = i;
        typeSelect.dataset.configType = 'type';
        layerTypes.forEach((t) => {
          const o = document.createElement('option');
          o.value = t;
          o.textContent = t;
          if (t === 'dense') o.selected = true;
          typeSelect.appendChild(o);
        });
        typeDiv.appendChild(typeLabel);
        typeDiv.appendChild(typeSelect);
        layerGroup.appendChild(typeDiv);
        const optionsDiv = document.createElement('div');
        optionsDiv.className = 'layer-options-container';
        optionsDiv.dataset.layerIndex = i;
        layerGroup.appendChild(optionsDiv);

        typeSelect.addEventListener(
          'change',
          () => (architectureTemplateSelect.value = 'custom')
        );

        const updateOptionsUI = (idx, selType) => {
          const optsDiv = hiddenLayersConfigContainer.querySelector(
            `.layer-options-container[data-layer-index='${idx}']`
          );
          if (!optsDiv) return;
          optsDiv.innerHTML = '';
          const createIn = (l, id, t, v, mn, st, cfg, nt = null, tt = null) => {
            const dv = document.createElement('div');
            dv.className = 'input-group';
            const lb = document.createElement('label');
            lb.textContent = l;
            lb.htmlFor = id;
            if (tt) lb.title = tt;
            const ip = document.createElement('input');
            ip.type = t;
            ip.id = id;
            ip.value = v;
            if (mn !== null) ip.min = mn;
            if (st !== null) ip.step = st;
            ip.dataset.layerIndex = idx;
            ip.dataset.configType = cfg;
            dv.appendChild(lb);
            dv.appendChild(ip);
            if (nt) {
              const p = document.createElement('p');
              p.className = 'layer-note';
              p.textContent = nt;
              dv.appendChild(p);
            }
            return dv;
          };
          const createSel = (l, id, opts, selV, cfg, tt = null) => {
            const dv = document.createElement('div');
            dv.className = 'input-group';
            const lb = document.createElement('label');
            lb.textContent = l;
            lb.htmlFor = id;
            if (tt) lb.title = tt;
            const sel = document.createElement('select');
            sel.id = id;
            sel.dataset.layerIndex = idx;
            sel.dataset.configType = cfg;
            opts.forEach((o) => {
              const op = document.createElement('option');
              op.value = o;
              op.textContent = o;
              if (o === selV) op.selected = true;
              sel.appendChild(op);
            });
            dv.appendChild(lb);
            dv.appendChild(sel);
            return dv;
          };
          const createChk = (l, id, chkd, cfg, tt = null) => {
            const dv = document.createElement('div');
            dv.className = 'input-group';
            const lb = document.createElement('label');
            const ip = document.createElement('input');
            ip.type = 'checkbox';
            ip.id = id;
            ip.checked = chkd;
            ip.dataset.layerIndex = idx;
            ip.dataset.configType = cfg;
            const sp = document.createElement('span');
            sp.textContent = l;
            if (tt) sp.title = tt;
            lb.appendChild(ip);
            lb.appendChild(sp);
            dv.appendChild(lb);
            return dv;
          };
          const createNt = (txt) => {
            const p = document.createElement('p');
            p.className = 'layer-note';
            p.textContent = txt;
            const dv = document.createElement('div');
            dv.style.gridColumn = '1/-1';
            dv.appendChild(p);
            return dv;
          };
          if (selType === 'dense') {
            optsDiv.appendChild(
              createIn(
                'Nodes:',
                `layerNodes_${idx}`,
                'number',
                10,
                1,
                1,
                'size',
                null,
                'Num neurons.'
              )
            );
            optsDiv.appendChild(
              createSel(
                'Activation:',
                `layerAct_${idx}`,
                activationTypes,
                'tanh',
                'activation',
                'Neuron output function.'
              )
            );
            optsDiv.appendChild(
              createChk(
                'Use Bias:',
                `layerBias_${idx}`,
                true,
                'useBias',
                'Add learnable bias term?'
              )
            );
          } else if (selType === 'attention') {
            optsDiv.appendChild(
              createIn(
                'Num Heads:',
                `layerHeads_${idx}`,
                'number',
                2,
                1,
                1,
                'numHeads',
                null,
                'Parallel attention mechanisms. Input size must be divisible by heads.'
              )
            );
            optsDiv.appendChild(
              createNt(
                'Input size must be divisible by Num Heads. Output size matches input.'
              )
            );
          } else if (selType === 'layernorm') {
            optsDiv.appendChild(
              createNt('Normalizes features across the feature dimension.')
            );
          } else if (selType === 'dropout') {
            optsDiv.appendChild(
              createIn(
                'Dropout Rate:',
                `layerRate_${idx}`,
                'number',
                0.5,
                0,
                0.01,
                'rate',
                'Fraction of neurons to zero out during training (0 to <1). Helps prevent overfitting.',
                'Higher value means more dropout.'
              )
            );
          } else if (selType === 'softmax') {
            optsDiv.appendChild(
              createNt(
                'Outputs probabilities summing to 1. For multi-class classification.'
              )
            );
          }
        };
        typeSelect.addEventListener('change', (event) =>
          updateOptionsUI(i, event.target.value)
        );
        hiddenLayersConfigContainer.appendChild(layerGroup);
        updateOptionsUI(i, typeSelect.value);
      }
    }
    numHiddenLayersInput.addEventListener('change', (event) => {
      const numLayers = Math.max(0, parseInt(event.target.value) || 0);
      event.target.value = numLayers;
      createLayerConfigUI(numLayers);

      architectureTemplateSelect.value = 'custom';
    });
    createLayerConfigUI(parseInt(numHiddenLayersInput.value));
    optimizerSelect.addEventListener('change', () => {
      decayRateGroup.style.display =
      optimizerSelect.value === 'rmsprop' ? 'block' : 'none';
    });
    decayRateGroup.style.display =
    optimizerSelect.value === 'rmsprop' ? 'block' : 'none';

    lrSchedulerSelect.addEventListener('change', () => {
      const selectedSchedule = lrSchedulerSelect.value;
      lrStepParamsDiv.style.display =
      selectedSchedule === 'step' ? 'grid' : 'none';
      lrExpParamsDiv.style.display =
      selectedSchedule === 'exponential' ? 'grid' : 'none';
    });
    lrSchedulerSelect.dispatchEvent(new Event('change'));

    architectureTemplateSelect.addEventListener('change', (event) => {
      const templateKey = event.target.value;
      if (templateKey === 'custom') {
        return;
      }

      const template = architectureTemplates[templateKey];
      if (!template) {
        console.warn('Selected template not found:', templateKey);
        return;
      }

      statsEl.innerHTML = `Applying '${templateKey}' template...`;
      numHiddenLayersInput.value = template.numHidden;
      numHiddenLayersInput.dispatchEvent(new Event('change'));

      setTimeout(() => {
        try {
          template.layers.forEach((layerConfig, i) => {
            const setV = (s, v) => {
              const e = hiddenLayersConfigContainer.querySelector(s);
              if (e && v !== undefined) e.value = v;
            };
            const setC = (s, c) => {
              const e = hiddenLayersConfigContainer.querySelector(s);
              if (e && c !== undefined) e.checked = c;
            };
            const triggerChange = (s) => {
              const e = hiddenLayersConfigContainer.querySelector(s);
              if (e) e.dispatchEvent(new Event('change', { bubbles: true }));
            };

            setV(
              `select[data-layer-index="${i}"][data-config-type="type"]`,
              layerConfig.type || 'dense'
            );
            triggerChange(
              `select[data-layer-index="${i}"][data-config-type="type"]`
            );

            switch (layerConfig.type) {
            case 'dense':
              setV(
                `input[data-layer-index="${i}"][data-config-type="size"]`,
                layerConfig.size
              );
              setV(
                `select[data-layer-index="${i}"][data-config-type="activation"]`,
                layerConfig.activation
              );
              setC(
                `input[data-layer-index="${i}"][data-config-type="useBias"]`,
                layerConfig.useBias
              );
              break;
            case 'attention':
              setV(
                `input[data-layer-index="${i}"][data-config-type="numHeads"]`,
                layerConfig.numHeads
              );
              break;
            case 'dropout':
              setV(
                `input[data-layer-index="${i}"][data-config-type="rate"]`,
                layerConfig.rate
              );
              break;
            }
          });

          const performanceIndicator = useOptimized ? 
            '<span style="color: #4CAF50; font-weight: bold;">🚀 Optimized</span>' : 
            '<span style="color: #FF9800; font-weight: bold;">📊 Original</span>';
          statsEl.innerHTML = `Applied <span style="color: #87CEEB; font-weight: bold;">${templateKey}</span> template. ${performanceIndicator} Ready.`;
        } catch (error) {
          console.error('Error applying template UI:', error);
          statsEl.innerHTML = `<span class="error">Error applying template: ${error.message}</span>`;
          architectureTemplateSelect.value = 'custom';
        }
      }, 20);
    });

    trainButton.addEventListener('click', async () => {
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
      try {
        const weightInitMethod = document.getElementById('weightInit').value;

        const outDims = parseInt(numOutputDimsInput.value) || 1;
        const trainingData = parseCSV(trainingDataTextarea.value, outDims);
        const testData = parseCSV(testDataTextarea.value, outDims);
        if (trainingData.length === 0)
          throw new Error('Training data empty/invalid.');
        if (!trainingData[0]?.input || !trainingData[0]?.output)
          throw new Error('Cannot get input/output size from data.');
        nn.reset();
        const numHidden = parseInt(numHiddenLayersInput.value);
        const layerCfgs = [];
        const numIns = trainingData[0].input.length;
        let curInSize = numIns;
        for (let i = 0; i < numHidden; i++) {
          const getV = (s) => hiddenLayersConfigContainer.querySelector(s)?.value;
          const getC = (s) =>
            hiddenLayersConfigContainer.querySelector(s)?.checked;
          const lType =
          getV(`select[data-layer-index="${i}"][data-config-type="type"]`) ||
          'dense';

          const cfg = {
            type: lType,
            inputSize: curInSize,
            weightInit: weightInitMethod
          };
          switch (lType) {
          case 'dense':
            cfg.outputSize = parseInt(
              getV(`input[data-layer-index="${i}"][data-config-type="size"]`) ||
                1
            );
            if (cfg.outputSize <= 0)
              throw new Error(`L${i + 1} Dense: Invalid nodes.`);
            cfg.activation =
              getV(
                `select[data-layer-index="${i}"][data-config-type="activation"]`
              ) || 'tanh';
            cfg.useBias =
              getC(
                `input[data-layer-index="${i}"][data-config-type="useBias"]`
              ) ?? true;
            break;
          case 'attention':
            cfg.numHeads = parseInt(
              getV(
                `input[data-layer-index="${i}"][data-config-type="numHeads"]`
              ) || 1
            );
            if (cfg.numHeads <= 0)
              throw new Error(`L${i + 1} Attn: Invalid heads.`);
            if (curInSize % cfg.numHeads !== 0)
              throw new Error(
                `L${i + 1} Attn: Input ${curInSize} not divisible by ${cfg.numHeads} heads.`
              );
            cfg.outputSize = curInSize;
            break;
          case 'dropout':
            cfg.rate = parseFloat(
              getV(`input[data-layer-index="${i}"][data-config-type="rate"]`) ||
                0
            );
            if (cfg.rate < 0 || cfg.rate >= 1)
              throw new Error(`L${i + 1} Dropout: Invalid rate.`);
            cfg.outputSize = curInSize;
            break;
          case 'layernorm':
          case 'softmax':
            cfg.outputSize = curInSize;
            break;
          default:
            throw new Error(`L${i + 1}: Unknown type "${lType}".`);
          }
          if (!cfg.outputSize) throw new Error(`L${i + 1}: No output size.`);
          layerCfgs.push(cfg);
          curInSize = cfg.outputSize;
        }

        const isAutoencoder = architectureTemplateSelect.value === 'autoencoder';
        const numOuts = isAutoencoder ? numIns : trainingData[0].output.length;
        if (numOuts <= 0) throw new Error('Zero output cols defined.');

        const selectedLoss = lossFunctionSelect.value;

        let finalAct = 'none';

        if (selectedLoss === 'crossentropy') {
          if (isAutoencoder) {
            finalAct =
            architectureTemplates['autoencoder']?.finalActivationHint ||
            'sigmoid';
            console.log(
              `Autoencoder with Cross-Entropy? Using template hint or sigmoid: ${finalAct}`
            );
          } else if (numOuts > 1) {
            finalAct = 'softmax';
          } else if (numOuts === 1) {
            finalAct = 'sigmoid';
          }
        } else if (selectedLoss === 'mse') {
          if (isAutoencoder) {
            finalAct =
            architectureTemplates['autoencoder']?.finalActivationHint ||
            'sigmoid';
            console.log(
              `Autoencoder with MSE, using template hint or sigmoid: ${finalAct}`
            );
          } else {
            finalAct = 'none';
            console.log('MSE loss selected, using linear final activation: none');
          }
        }

        console.log(
          `Adding final dense: ${curInSize}->${numOuts} (Act:${finalAct})`
        );
        layerCfgs.push({
          type: 'dense',
          inputSize: curInSize,
          outputSize: numOuts,
          activation: finalAct,
          useBias: true
        });

        layerCfgs.forEach((cfg, i) => {
          try {
            nn.layer(cfg);
          } catch (e) {
            throw new Error(`Cfg L${i + 1}(${cfg.type}): ${e.message}`);
          }
        });
        if (nn.layers.length === 0) throw new Error('Zero layers configured.');
        if (nn.debug) console.log('Net structure:', nn.layers);
        const opts = {
          epochs: parseInt(epochsInput.value) || 50,
          learningRate: parseFloat(learningRateInput.value) || 0.01,
          batchSize: parseInt(batchSizeInput.value) || 8,
          testSet: testData.length > 0 ? testData : null,
          optimizer: optimizerSelect.value,
          lossFunction: lossFunctionSelect.value,
          l2Lambda: parseFloat(l2LambdaInput.value) || 0,
          decayRate: parseFloat(decayRateInput.value) || 0.9,
          gradientClipValue: parseFloat(gradientClipValueInput.value) || 0,
          usePositionalEncoding: usePositionalEncodingCheckbox.checked,
          lrSchedule: lrSchedulerSelect.value,
          lrStepDecayFactor:
          parseFloat(document.getElementById('lrStepDecayFactor').value) || 0.1,
          lrStepDecaySize:
          parseInt(document.getElementById('lrStepDecaySize').value) || 10,
          lrExpDecayRate:
          parseFloat(document.getElementById('lrExpDecayRate').value) || 0.95,
          callback: async (
            ep,
            trL,
            tstL,
            metricName,
            metricVal,
            lastForwardCache
          ) => {
            lossHistory.push({ train: trL, test: tstL });
            drawLossGraph();
            epochBar.style.width = `${(ep / opts.epochs) * 100}%`;



            const currentLR = nn.getCurrentLearningRate(
              ep - 1,
              opts.learningRate,
              opts
            );
            const lrString =
            opts.lrSchedule !== 'none'
              ? ` | LR: ${currentLR.toExponential(2)}`
              : '';

            let statusText = `Ep:${ep}/${opts.epochs} | Loss:${trL.toFixed(6)}`;
            if (tstL !== null) {
              statusText += ` | Val Loss:${tstL.toFixed(6)}`;
            }

            if (metricName && metricVal !== null && !isNaN(metricVal)) {
              statusText += ` | Val ${metricName}:${metricVal.toFixed(4)}`;
            }
            statusText += lrString;
            statsEl.innerHTML = statusText;

            if (
              lastForwardCache &&
            lastForwardCache.activations &&
            lastForwardCache.activations.length > 0
            ) {
              nn.lastActivations = lastForwardCache.activations;

              if (!window._drawNetworkScheduled) {
                window._drawNetworkScheduled = true;
                requestAnimationFrame(() => {
                  try {
                    drawNetwork();
                  } catch (drawErr) {
                    console.warn('Error during dynamic drawNetwork:', drawErr);
                  } finally {
                    window._drawNetworkScheduled = false;
                  }
                });
              }
            }
          }
        };
        statsEl.innerHTML =
        `Training (${opts.optimizer}, ${opts.lossFunction}` +
        (opts.lrSchedule !== 'none' ? `, ${opts.lrSchedule} LR` : '') +
        ')...';

        window._drawNetworkScheduled = false;

        const startTime = performance.now();
        const summary = await nn.train(trainingData, opts);
        const endTime = performance.now();
        const trainingTime = endTime - startTime;
        const totalParams = nn.getTotalParameters();
      
        const performanceIndicator = useOptimized ? 
          '<span style="color: #4CAF50; font-weight: bold;">🚀 Optimized</span>' : 
          '<span style="color: #FF9800; font-weight: bold;">📊 Original</span>';
      
        statsEl.innerHTML =
        `<strong>Done!</strong> Loss:${summary.trainLoss.toFixed(6)}` +
        (summary.testLoss !== null
          ? `, Val:${summary.testLoss.toFixed(6)}`
          : '') +
        ` | Params:${totalParams.toLocaleString()} | ${performanceIndicator} | Time:${trainingTime.toFixed(0)}ms`;
        console.log('Final Summary:', summary);
        if (trainingData.length > 0) {
          try {
            nn.predict(trainingData[0].input);
            drawNetwork();
          } catch (e) {}
        }
      } catch (error) {
        console.error('Train err:', error);
        statsEl.innerHTML = `<span class="error">Error: ${error.message}</span>`;
      } finally {
        trainButton.disabled = false;
        trainButton.textContent = 'Train Model';
        predictButton.disabled = false;
        saveButton.disabled = false;
        loadButton.disabled = false;
        unloadButton.disabled = false;
        pauseButton.disabled = true;
        resumeButton.disabled = true;
      }
    });

    pauseButton.addEventListener('click', () => {
      nn.pauseTraining();
      pauseButton.disabled = true;
      resumeButton.disabled = false;
      statsEl.innerHTML = 'Training paused';
    });

    resumeButton.addEventListener('click', () => {
      nn.resumeTraining();
      pauseButton.disabled = false;
      resumeButton.disabled = true;
    });



    function drawNetworkLocal() {
      if (!networkCtx || !networkCanvas) return;
      const networkContainer = networkCanvas.parentElement;
      if (!networkContainer) return;
      const containerWidth = networkContainer.clientWidth;
      const containerHeight = networkContainer.clientHeight;
      
      drawNetwork(networkCtx, networkCanvas, nn, containerWidth, containerHeight);
    }
      const hasModel =
      nn.lastActivations &&
      nn.lastActivations.length > 0 &&
      nn.layers &&
      nn.layers.length > 0;

      if (!hasModel) {
        networkCtx.fillStyle = '#555';
        networkCtx.font = '10px monospace';
        networkCtx.textAlign = 'center';
        networkCtx.textBaseline = 'middle';

        if (networkCanvas.width !== containerWidth)
          networkCanvas.width = containerWidth;
        if (networkCanvas.height !== containerHeight)
          networkCanvas.height = containerHeight;
        networkCtx.fillText(
          'Train/Predict to visualize',
          containerWidth / 2,
          containerHeight / 2
        );
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
        lblFnt = '10px monospace',
        lblClr = '#aaa';
      const nVizLyrs = nn.lastActivations.length;
      const baseSpacing = 150;
      const minLayerSpacing = Math.max(
        120,
        Math.min(baseSpacing, containerWidth / (nVizLyrs > 1 ? nVizLyrs : 1))
      );

      const requiredWidth =
      nVizLyrs <= 1
        ? containerWidth
        : pad * 2 + (nVizLyrs - 1) * minLayerSpacing;

      const canvasDrawWidth = Math.max(containerWidth, requiredWidth * 1.2);

      networkCanvas.width = canvasDrawWidth;
      networkCanvas.height = containerHeight;

      const drawAreaWidth = canvasDrawWidth - pad * 2;
      const drawAreaHeight = containerHeight - pad * 2;
      const layerXs = Array.from(
        { length: nVizLyrs },
        (_, i) =>
          pad +
        (nVizLyrs === 1
          ? drawAreaWidth / 2
          : (drawAreaWidth * i) / (nVizLyrs - 1))
      );

      const layerPos = [];
      nn.lastActivations.forEach((act, lIdx) => {
        if (!act || typeof act.length !== 'number') {
          if (nn.debug)
            console.warn(
              `drawNetwork L${lIdx}: Activation data is not array-like.`,
              act
            );
          layerPos.push([]);
          return;
        }

        const lNodes = [];
        const nNodes = act.length;
        const dNodes = Math.min(nNodes, maxNds);
        const lX = layerXs[lIdx];
        for (let j = 0; j < dNodes; j++) {
          const origIdx =
          nNodes <= maxNds ? j : Math.floor((j * nNodes) / dNodes);
          const nodeVal = act[origIdx];
          const nY =
          pad +
          (dNodes === 1
            ? drawAreaHeight / 2
            : (drawAreaHeight * j) / (dNodes - 1));

          lNodes.push({
            x: lX,
            y: nY,
            value: typeof nodeVal === 'number' && isFinite(nodeVal) ? nodeVal : 0
          });
        }
        if (nNodes > maxNds) {
          lNodes.push({
            x: lX,
            y: pad + drawAreaHeight + ellOff,
            value: 0,
            isEllipsis: true,
            originalCount: nNodes
          });
        }
        layerPos.push(lNodes);
      });

      networkCtx.lineWidth = 1;
      for (let i = 0; i < nVizLyrs - 1; i++) {
        const curNodes = layerPos[i].filter((n) => !n.isEllipsis);
        const nextNodes = layerPos[i + 1].filter((n) => !n.isEllipsis);
        const cfg = nn.layers[i];
        if (!cfg) continue;

        const isDenseW =
        cfg.type === 'dense' && nn.weights?.[i] instanceof Float32Array;
        const w = isDenseW ? nn.weights[i] : null;
        const inputSizeForLayer = cfg.inputSize;

        for (let j = 0; j < curNodes.length; j++) {
          for (let k = 0; k < nextNodes.length; k++) {
            let op = 0.1,
              col = '100,100,100',
              lw = 0.5;
            let lineDash = [];
            let weight = null;

            if (
              isDenseW &&
            w &&
            typeof inputSizeForLayer === 'number' &&
            inputSizeForLayer > 0
            ) {
              const weightIndex = k * inputSizeForLayer + j;
              if (weightIndex >= 0 && weightIndex < w.length) {
                weight = w[weightIndex];
              } else {
                if (nn.debug)
                  console.warn(
                    `drawNetwork L${i} Dense: Invalid weight index ${weightIndex} (nextIdx=${k}, currIdx=${j}, inSize=${inputSizeForLayer}, wLen=${w.length})`
                  );
                weight = null;
              }
            }

            if (
              isDenseW &&
            weight !== null &&
            typeof weight === 'number' &&
            typeof curNodes[j].value === 'number'
            ) {
              const wMag = Math.tanh(Math.abs(weight));
              const aMag = Math.tanh(Math.abs(curNodes[j].value));
              const combSig = wMag * 0.8 + aMag * 0.2;
              op = Math.min(Math.max(cBOp + combSig * (cMOp - cBOp), cBOp), cMOp);
              col = weight >= 0 ? '255,255,255' : '180,180,255';
              lw = Math.min(Math.max(0.5, op * cWScale), 2);
              lineDash = [];
            } else if (cfg.type === 'attention') {
              op = 0.5;
              lw = 1.2;
              col = '0,200,200';
              lineDash = [];
            } else if (cfg.type === 'layernorm') {
              op = 0.4;
              lw = 0.8;
              col = '255,255,0';
              lineDash = [2, 2];
            } else if (cfg.type === 'dropout') {
              op = 0.2;
              lw = 0.6;
              col = '255,165,0';
              lineDash = [3, 3];
            } else if (cfg.type === 'softmax') {
              op = 0.4;
              lw = 1.0;
              col = '200,0,200';
              lineDash = [];
            } else if (!isDenseW) {
              op = 0.1;
              lw = 0.5;
              col = '100,100,100';
              lineDash = [];
            } else {
              op = 0.05;
              lw = 0.3;
              col = '80,80,80';
              lineDash = [1, 2];
            }

            networkCtx.strokeStyle = `rgba(${col},${op})`;
            networkCtx.lineWidth = lw;
            networkCtx.setLineDash(lineDash || []);
            networkCtx.beginPath();
            networkCtx.moveTo(curNodes[j].x, curNodes[j].y);
            networkCtx.lineTo(nextNodes[k].x, nextNodes[k].y);
            networkCtx.stroke();
            networkCtx.setLineDash([]);
          }
        }
      }

      networkCtx.textAlign = 'center';

      if (layerXs.length > 2) {
        const firstHiddenLayerX = layerXs[1];
        const lastHiddenLayerX = layerXs[layerXs.length - 2];
        const centerX = (firstHiddenLayerX + lastHiddenLayerX) / 2;

        networkCtx.fillStyle = lblClr;
        networkCtx.font = lblFnt;
        networkCtx.textBaseline = 'bottom';
        networkCtx.fillText('Layers', centerX, pad - lblOff / 2);
      }

      layerPos.forEach((lNodes, lIdx) => {
        networkCtx.fillStyle = lblClr;
        networkCtx.font = lblFnt;
        networkCtx.textBaseline = 'bottom';
        if (lIdx === 0) {
          networkCtx.fillText('Input', layerXs[lIdx], pad - lblOff / 2);
        } else if (lIdx === layerPos.length - 1) {
          networkCtx.fillText('Output', layerXs[lIdx], pad - lblOff / 2);
        }

        lNodes.forEach((n) => {
          if (n.isEllipsis) {
            networkCtx.fillStyle = '#777';
            networkCtx.font = '10px monospace';
            networkCtx.textBaseline = 'top';
            networkCtx.fillText(`(${n.originalCount} nodes)`, n.x, n.y);
          } else {
            const actStr = Math.tanh(Math.abs(n.value));
            const r = nRBase + actStr * nRScale;
            const op = 0.3 + actStr * 0.7;
            const col = n.value >= 0 ? '255,255,255' : '200,200,255';
            networkCtx.fillStyle = `rgba(${col},${op})`;
            networkCtx.strokeStyle = 'rgba(255,255,255,0.6)';
            networkCtx.lineWidth = 1;
            networkCtx.beginPath();
            networkCtx.arc(n.x, n.y, r, 0, Math.PI * 2);
            networkCtx.fill();
            networkCtx.stroke();
          }
        });
      });
    }
    function resizeCanvases() {
      const lossContainer = lossCanvas?.parentElement;

      const networkContainer = document.getElementById('network-viz-container');

      if (lossContainer?.clientWidth > 0 && lossCanvas) {
        lossCanvas.width = lossContainer.clientWidth;
        lossCanvas.height = lossContainer.clientHeight;
        drawLossGraph();
      }

      if (networkContainer?.clientWidth > 0 && networkCanvas) {
        drawNetwork();
      }
    }
    window.addEventListener('resize', resizeCanvases);
    setTimeout(resizeCanvases, 150);

    saveButton.addEventListener('click', () => {
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
    });

    loadButton.addEventListener('click', () => {
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
    });
    predictButton.addEventListener('click', () => {
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
    });
    unloadButton.addEventListener('click', () => {
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
    });
  });
}

