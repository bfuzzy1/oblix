import { performance } from 'perf_hooks';
import { Oblix } from '../src/network.js';
import { OptimizedOblix } from '../src/optimized/network.js';

export async function run() {
  console.log('🔬 ORIGINAL vs OPTIMIZED NEURAL NETWORK COMPARISON\n');
  
  const results = {
    initialization: {},
    training: {},
    prediction: {},
    memory: {},
    overall: {}
  };

  // Test 1: Network Initialization Performance
  console.log('📊 Testing Network Initialization...');
  await benchmarkInitialization(results);
  
  // Test 2: Training Performance
  console.log('📊 Testing Training Performance...');
  await benchmarkTraining(results);
  
  // Test 3: Prediction Performance
  console.log('📊 Testing Prediction Performance...');
  await benchmarkPrediction(results);
  
  // Test 4: Memory Usage Comparison
  console.log('📊 Testing Memory Usage...');
  await benchmarkMemoryUsage(results);
  
  // Generate comprehensive report
  generateComparisonReport(results);
}

async function benchmarkInitialization(results) {
  const networkSizes = [
    { name: 'small', layers: [10, 5, 1] },
    { name: 'medium', layers: [50, 25, 10, 1] },
    { name: 'large', layers: [100, 50, 25, 10, 1] }
  ];
  
  results.initialization = {};
  
  for (const size of networkSizes) {
    console.log(`  Testing ${size.name} network...`);
    
    // Test original initialization
    const originalStart = performance.now();
    const originalNet = new Oblix(false);
    let currentSize = 2; // Input size
    for (let i = 0; i < size.layers.length; i++) {
      originalNet.layer({
        type: 'dense',
        inputSize: currentSize,
        outputSize: size.layers[i],
        activation: 'relu',
        useBias: true
      });
      currentSize = size.layers[i];
    }
    const originalTime = performance.now() - originalStart;
    
    // Test optimized initialization
    const optimizedStart = performance.now();
    const optimizedNet = new OptimizedOblix(false);
    currentSize = 2; // Input size
    for (let i = 0; i < size.layers.length; i++) {
      optimizedNet.layer({
        type: 'dense',
        inputSize: currentSize,
        outputSize: size.layers[i],
        activation: 'relu',
        useBias: true
      });
      currentSize = size.layers[i];
    }
    const optimizedTime = performance.now() - optimizedStart;
    
    const speedup = originalTime / optimizedTime;
    
    results.initialization[size.name] = {
      original: originalTime,
      optimized: optimizedTime,
      speedup,
      parameters: originalNet.getTotalParameters()
    };
    
    console.log(`    ${size.name}: ${speedup.toFixed(2)}x speedup`);
  }
}

async function benchmarkTraining(results) {
  // Create training data
  const trainingData = [];
  for (let i = 0; i < 100; i++) {
    trainingData.push({
      input: [Math.random(), Math.random()],
      output: [Math.random()]
    });
  }
  
  const testData = trainingData.slice(0, 20);
  
  results.training = {};
  
  // Test original training
  console.log('  Testing original training...');
  const originalNet = new Oblix(false);
  originalNet.layer({ type: 'dense', inputSize: 2, outputSize: 10, activation: 'relu' });
  originalNet.layer({ type: 'dense', inputSize: 10, outputSize: 5, activation: 'relu' });
  originalNet.layer({ type: 'dense', inputSize: 5, outputSize: 1, activation: 'tanh' });
  
  const originalStart = performance.now();
  const originalResult = await originalNet.train(trainingData, {
    epochs: 10,
    learningRate: 0.01,
    batchSize: 8,
    testSet: testData
  });
  const originalTime = performance.now() - originalStart;
  
  // Test optimized training
  console.log('  Testing optimized training...');
  const optimizedNet = new OptimizedOblix(false);
  optimizedNet.layer({ type: 'dense', inputSize: 2, outputSize: 10, activation: 'relu' });
  optimizedNet.layer({ type: 'dense', inputSize: 10, outputSize: 5, activation: 'relu' });
  optimizedNet.layer({ type: 'dense', inputSize: 5, outputSize: 1, activation: 'tanh' });
  
  const optimizedStart = performance.now();
  const optimizedResult = await optimizedNet.train(trainingData, {
    epochs: 10,
    learningRate: 0.01,
    batchSize: 8,
    testSet: testData
  });
  const optimizedTime = performance.now() - optimizedStart;
  
  const speedup = originalTime / optimizedTime;
  
  results.training = {
    original: {
      time: originalTime,
      finalLoss: originalResult.trainLoss,
      testLoss: originalResult.testLoss
    },
    optimized: {
      time: optimizedTime,
      finalLoss: optimizedResult.trainLoss,
      testLoss: optimizedResult.testLoss
    },
    speedup,
    lossDifference: Math.abs(originalResult.trainLoss - optimizedResult.trainLoss)
  };
  
  console.log(`    Training: ${speedup.toFixed(2)}x speedup`);
  console.log(`    Loss difference: ${results.training.lossDifference.toFixed(6)}`);
}

async function benchmarkPrediction(results) {
  // Create networks
  const originalNet = new Oblix(false);
  originalNet.layer({ type: 'dense', inputSize: 2, outputSize: 10, activation: 'relu' });
  originalNet.layer({ type: 'dense', inputSize: 10, outputSize: 1, activation: 'tanh' });
  
  const optimizedNet = new OptimizedOblix(false);
  optimizedNet.layer({ type: 'dense', inputSize: 2, outputSize: 10, activation: 'relu' });
  optimizedNet.layer({ type: 'dense', inputSize: 10, outputSize: 1, activation: 'tanh' });
  
  // Create test inputs
  const testInputs = [];
  for (let i = 0; i < 1000; i++) {
    testInputs.push([Math.random(), Math.random()]);
  }
  
  results.prediction = {};
  
  // Test original prediction
  console.log('  Testing original prediction...');
  const originalStart = performance.now();
  const originalPredictions = [];
  for (const input of testInputs) {
    originalPredictions.push(originalNet.predict(input));
  }
  const originalTime = performance.now() - originalStart;
  
  // Test optimized prediction
  console.log('  Testing optimized prediction...');
  const optimizedStart = performance.now();
  const optimizedPredictions = [];
  for (const input of testInputs) {
    optimizedPredictions.push(optimizedNet.predict(input));
  }
  const optimizedTime = performance.now() - optimizedStart;
  
  // Verify predictions are similar
  let maxDiff = 0;
  for (let i = 0; i < originalPredictions.length; i++) {
    const diff = Math.abs(originalPredictions[i][0] - optimizedPredictions[i][0]);
    maxDiff = Math.max(maxDiff, diff);
  }
  
  const speedup = originalTime / optimizedTime;
  const isValid = maxDiff < 1e-6;
  
  results.prediction = {
    original: originalTime,
    optimized: optimizedTime,
    speedup,
    maxDifference: maxDiff,
    isValid
  };
  
  console.log(`    Prediction: ${speedup.toFixed(2)}x speedup, ${isValid ? '✅' : '❌'} accuracy`);
}

async function benchmarkMemoryUsage(results) {
  const { memoryMonitor } = await import('../src/optimized/memory.js');
  
  results.memory = {};
  
  // Test memory usage during network operations
  console.log('  Testing memory usage...');
  
  // Original network memory usage
  memoryMonitor.takeSnapshot('before_original');
  const originalNet = new Oblix(false);
  for (let i = 0; i < 5; i++) {
    originalNet.layer({ type: 'dense', inputSize: 10, outputSize: 10, activation: 'relu' });
  }
  memoryMonitor.takeSnapshot('after_original');
  
  // Optimized network memory usage
  memoryMonitor.takeSnapshot('before_optimized');
  const optimizedNet = new OptimizedOblix(false);
  for (let i = 0; i < 5; i++) {
    optimizedNet.layer({ type: 'dense', inputSize: 10, outputSize: 10, activation: 'relu' });
  }
  memoryMonitor.takeSnapshot('after_optimized');
  
  const memoryTrend = memoryMonitor.getMemoryTrend();
  
  results.memory = {
    originalGrowth: memoryTrend?.totalGrowth || 0,
    optimizedGrowth: memoryTrend?.totalGrowth || 0,
    improvement: memoryTrend ? (memoryTrend.totalGrowth > 0 ? 'reduced' : 'increased') : 'unknown'
  };
  
  console.log(`    Memory usage: ${results.memory.improvement}`);
}

function generateComparisonReport(results) {
  console.log('\n📈 ORIGINAL vs OPTIMIZED COMPARISON REPORT');
  console.log('=============================================\n');
  
  // Calculate overall improvements
  let totalSpeedup = 0;
  let testCount = 0;
  
  // Initialization summary
  console.log('🎯 NETWORK INITIALIZATION:');
  for (const [size, result] of Object.entries(results.initialization)) {
    console.log(`  ${size}: ${result.speedup.toFixed(2)}x speedup (${result.parameters} params)`);
    totalSpeedup += result.speedup;
    testCount++;
  }
  
  // Training summary
  console.log('\n🔢 TRAINING PERFORMANCE:');
  const training = results.training;
  console.log(`  Speedup: ${training.speedup.toFixed(2)}x`);
  console.log(`  Original time: ${training.original.time.toFixed(2)}ms`);
  console.log(`  Optimized time: ${training.optimized.time.toFixed(2)}ms`);
  console.log(`  Loss difference: ${training.lossDifference.toFixed(6)}`);
  
  totalSpeedup += training.speedup;
  testCount++;
  
  // Prediction summary
  console.log('\n🎯 PREDICTION PERFORMANCE:');
  const prediction = results.prediction;
  console.log(`  Speedup: ${prediction.speedup.toFixed(2)}x`);
  console.log(`  Accuracy: ${prediction.isValid ? '✅ Valid' : '❌ Invalid'}`);
  console.log(`  Max difference: ${prediction.maxDifference.toFixed(6)}`);
  
  totalSpeedup += prediction.speedup;
  testCount++;
  
  // Memory summary
  console.log('\n💾 MEMORY USAGE:');
  const memory = results.memory;
  console.log(`  Memory improvement: ${memory.improvement}`);
  
  // Overall summary
  console.log('\n🏆 OVERALL PERFORMANCE ASSESSMENT:');
  if (testCount > 0) {
    const overallSpeedup = totalSpeedup / testCount;
    console.log(`  Average speedup: ${overallSpeedup.toFixed(2)}x`);
    
    if (overallSpeedup > 2.0) {
      console.log('  🎉 EXCELLENT: Outstanding performance improvements!');
    } else if (overallSpeedup > 1.5) {
      console.log('  ✅ VERY GOOD: Significant performance improvements!');
    } else if (overallSpeedup > 1.1) {
      console.log('  ✅ GOOD: Moderate performance improvements!');
    } else {
      console.log('  ⚠️  MINIMAL: Performance improvements are minimal');
    }
  }
  
  // Detailed results
  console.log('\n📊 DETAILED COMPARISON RESULTS:');
  console.log(JSON.stringify(results, null, 2));
}