import { performance } from 'perf_hooks';
import { oblixActivations } from '../src/activations.js';
import { optimizedMath } from '../src/optimized/math.js';
import { NeuralNetworkWorker } from '../src/optimized/workers.js';
import { memoryOptimizedOps, memoryMonitor } from '../src/optimized/memory.js';

export async function runPerformanceBenchmarks() {
  console.log('🚀 Starting Performance Optimization Benchmarks\n');
  
  const results = {
    activationFunctions: {},
    matrixOperations: {},
    memoryUsage: {},
    workerOperations: {},
    overall: {}
  };

  // Test 1: Activation Functions Performance
  console.log('📊 Testing Activation Functions...');
  await benchmarkActivationFunctions(results);
  
  // Test 2: Matrix Operations Performance
  console.log('📊 Testing Matrix Operations...');
  await benchmarkMatrixOperations(results);
  
  // Test 3: Memory Usage Performance
  console.log('📊 Testing Memory Usage...');
  await benchmarkMemoryUsage(results);
  
  // Test 4: Web Worker Performance
  console.log('📊 Testing Web Worker Operations...');
  await benchmarkWorkerOperations(results);
  
  // Generate summary report
  generateReport(results);
}

async function benchmarkActivationFunctions(results) {
  const activations = ['tanh', 'sigmoid', 'relu', 'leakyrelu', 'gelu'];
  const sampleSizes = [1000, 10000, 100000];
  
  for (const activation of activations) {
    results.activationFunctions[activation] = {};
    
    for (const size of sampleSizes) {
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        data[i] = (Math.random() - 0.5) * 4; // Range: -2 to 2
      }
      
      // Test original implementation
      const originalStart = performance.now();
      const originalResult = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        originalResult[i] = oblixActivations.apply(data[i], activation);
      }
      const originalTime = performance.now() - originalStart;
      
      // Test optimized implementation
      const optimizedStart = performance.now();
      const optimizedResult = optimizedMath.batchActivation(data, activation);
      const optimizedTime = performance.now() - optimizedStart;
      
      // Verify results are similar (within tolerance)
      let maxDiff = 0;
      for (let i = 0; i < size; i++) {
        const diff = Math.abs(originalResult[i] - optimizedResult[i]);
        maxDiff = Math.max(maxDiff, diff);
      }
      
      const speedup = originalTime / optimizedTime;
      const isValid = maxDiff < 1e-6;
      
      results.activationFunctions[activation][size] = {
        original: originalTime,
        optimized: optimizedTime,
        speedup,
        maxDifference: maxDiff,
        isValid
      };
      
      console.log(`  ${activation} (${size}): ${speedup.toFixed(2)}x speedup, ${isValid ? '✅' : '❌'} accuracy`);
    }
  }
}

async function benchmarkMatrixOperations(results) {
  const matrixSizes = [
    { rows: 10, cols: 10 },
    { rows: 50, cols: 50 },
    { rows: 100, cols: 100 },
    { rows: 200, cols: 200 }
  ];
  
  results.matrixOperations = {};
  
  for (const size of matrixSizes) {
    const { rows, cols } = size;
    const a = new Float32Array(rows * cols);
    const b = new Float32Array(cols * rows);
    
    // Initialize with random data
    for (let i = 0; i < a.length; i++) {
      a[i] = (Math.random() - 0.5) * 2;
    }
    for (let i = 0; i < b.length; i++) {
      b[i] = (Math.random() - 0.5) * 2;
    }
    
    // Test original matrix multiplication (simulated)
    const originalStart = performance.now();
    const originalResult = new Float32Array(rows * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < rows; j++) {
        let sum = 0;
        for (let k = 0; k < cols; k++) {
          sum += a[i * cols + k] * b[k * rows + j];
        }
        originalResult[i * rows + j] = sum;
      }
    }
    const originalTime = performance.now() - originalStart;
    
    // Test optimized matrix multiplication
    const optimizedStart = performance.now();
    const optimizedResult = optimizedMath.matrixMultiply(a, b, rows, cols, rows);
    const optimizedTime = performance.now() - optimizedStart;
    
    // Verify results
    let maxDiff = 0;
    for (let i = 0; i < originalResult.length; i++) {
      const diff = Math.abs(originalResult[i] - optimizedResult[i]);
      maxDiff = Math.max(maxDiff, diff);
    }
    
    const speedup = originalTime / optimizedTime;
    const isValid = maxDiff < 1e-6;
    
    results.matrixOperations[`${rows}x${cols}`] = {
      original: originalTime,
      optimized: optimizedTime,
      speedup,
      maxDifference: maxDiff,
      isValid
    };
    
    console.log(`  ${rows}x${cols}: ${speedup.toFixed(2)}x speedup, ${isValid ? '✅' : '❌'} accuracy`);
  }
}

async function benchmarkMemoryUsage(results) {
  const operations = [1000, 5000, 10000];
  
  results.memoryUsage = {};
  
  for (const count of operations) {
    // Test memory allocation patterns
    memoryMonitor.takeSnapshot('before_allocations');
    
    // Original approach: new allocations each time
    const originalStart = performance.now();
    const originalArrays = [];
    for (let i = 0; i < count; i++) {
      originalArrays.push(new Float32Array(1024));
    }
    const originalTime = performance.now() - originalStart;
    
    memoryMonitor.takeSnapshot('after_original');
    
    // Optimized approach: reuse from pool
    const optimizedStart = performance.now();
    const optimizedArrays = [];
    for (let i = 0; i < count; i++) {
      optimizedArrays.push(memoryOptimizedOps.createArray(1024));
    }
    const optimizedTime = performance.now() - optimizedStart;
    
    memoryMonitor.takeSnapshot('after_optimized');
    
    // Return arrays to pool
    for (const arr of optimizedArrays) {
      memoryOptimizedOps.returnArray(arr);
    }
    
    const memoryTrend = memoryMonitor.getMemoryTrend();
    
    results.memoryUsage[count] = {
      original: originalTime,
      optimized: optimizedTime,
      speedup: originalTime / optimizedTime,
      memoryGrowth: memoryTrend?.totalGrowth || 0
    };
    
    console.log(`  ${count} allocations: ${(originalTime / optimizedTime).toFixed(2)}x speedup`);
  }
}

async function benchmarkWorkerOperations(results) {
  const worker = new NeuralNetworkWorker();
  const initialized = await worker.initialize();
  
  if (!initialized) {
    console.log('  ⚠️  Web Workers not supported, skipping worker benchmarks');
    results.workerOperations = { supported: false };
    return;
  }
  
  const operations = [
    { type: 'batch_activation', data: { inputs: new Float32Array(10000).fill(0.5), activationType: 'tanh' } },
    { type: 'vector_operations', data: { operation: 'add', vectors: [new Float32Array(10000).fill(1), new Float32Array(10000).fill(2)], scalar: null } },
    { type: 'gradient_computation', data: { weights: new Float32Array(10000).fill(0.1), gradients: new Float32Array(10000).fill(0.01), learningRate: 0.01 } }
  ];
  
  results.workerOperations = { supported: true, operations: {} };
  
  for (const op of operations) {
    // Test main thread execution
    const mainStart = performance.now();
    let mainResult;
    
    switch (op.type) {
      case 'batch_activation':
        mainResult = optimizedMath.batchActivation(op.data.inputs, op.data.activationType);
        break;
      case 'vector_operations':
        mainResult = optimizedMath.vectorAdd(op.data.vectors[0], op.data.vectors[1]);
        break;
      case 'gradient_computation':
        mainResult = optimizedMath.computeGradients(op.data.weights, op.data.gradients, op.data.learningRate);
        break;
    }
    const mainTime = performance.now() - mainStart;
    
    // Test worker execution
    const workerStart = performance.now();
    const workerResult = await worker.execute(op.type, op.data);
    const workerTime = performance.now() - workerStart;
    
    // Verify results
    let maxDiff = 0;
    for (let i = 0; i < mainResult.length; i++) {
      const diff = Math.abs(mainResult[i] - workerResult[i]);
      maxDiff = Math.max(maxDiff, diff);
    }
    
    const speedup = mainTime / workerTime;
    const isValid = maxDiff < 1e-6;
    
    results.workerOperations.operations[op.type] = {
      mainThread: mainTime,
      worker: workerTime,
      speedup,
      maxDifference: maxDiff,
      isValid
    };
    
    console.log(`  ${op.type}: ${speedup.toFixed(2)}x speedup, ${isValid ? '✅' : '❌'} accuracy`);
  }
  
  worker.terminate();
}

function generateReport(results) {
  console.log('\n📈 PERFORMANCE OPTIMIZATION REPORT');
  console.log('=====================================\n');
  
  // Calculate overall improvements
  let totalSpeedup = 0;
  let testCount = 0;
  
  // Activation functions summary
  console.log('🎯 ACTIVATION FUNCTIONS:');
  for (const [activation, sizes] of Object.entries(results.activationFunctions)) {
    let avgSpeedup = 0;
    let validTests = 0;
    
    for (const [size, result] of Object.entries(sizes)) {
      if (result.isValid) {
        avgSpeedup += result.speedup;
        validTests++;
      }
    }
    
    if (validTests > 0) {
      avgSpeedup /= validTests;
      totalSpeedup += avgSpeedup;
      testCount++;
      console.log(`  ${activation}: ${avgSpeedup.toFixed(2)}x average speedup`);
    }
  }
  
  // Matrix operations summary
  console.log('\n🔢 MATRIX OPERATIONS:');
  let matrixAvgSpeedup = 0;
  let matrixValidTests = 0;
  
  for (const [size, result] of Object.entries(results.matrixOperations)) {
    if (result.isValid) {
      matrixAvgSpeedup += result.speedup;
      matrixValidTests++;
    }
  }
  
  if (matrixValidTests > 0) {
    matrixAvgSpeedup /= matrixValidTests;
    totalSpeedup += matrixAvgSpeedup;
    testCount++;
    console.log(`  Average: ${matrixAvgSpeedup.toFixed(2)}x speedup`);
  }
  
  // Memory usage summary
  console.log('\n💾 MEMORY USAGE:');
  let memoryAvgSpeedup = 0;
  let memoryTests = 0;
  
  for (const [count, result] of Object.entries(results.memoryUsage)) {
    memoryAvgSpeedup += result.speedup;
    memoryTests++;
  }
  
  if (memoryTests > 0) {
    memoryAvgSpeedup /= memoryTests;
    totalSpeedup += memoryAvgSpeedup;
    testCount++;
    console.log(`  Average: ${memoryAvgSpeedup.toFixed(2)}x speedup`);
  }
  
  // Worker operations summary
  console.log('\n🔄 WEB WORKER OPERATIONS:');
  if (results.workerOperations.supported) {
    let workerAvgSpeedup = 0;
    let workerValidTests = 0;
    
    for (const [op, result] of Object.entries(results.workerOperations.operations)) {
      if (result.isValid) {
        workerAvgSpeedup += result.speedup;
        workerValidTests++;
      }
    }
    
    if (workerValidTests > 0) {
      workerAvgSpeedup /= workerValidTests;
      totalSpeedup += workerAvgSpeedup;
      testCount++;
      console.log(`  Average: ${workerAvgSpeedup.toFixed(2)}x speedup`);
    }
  } else {
    console.log('  Not supported in this environment');
  }
  
  // Overall summary
  console.log('\n🏆 OVERALL PERFORMANCE IMPROVEMENTS:');
  if (testCount > 0) {
    const overallSpeedup = totalSpeedup / testCount;
    console.log(`  Average speedup across all optimizations: ${overallSpeedup.toFixed(2)}x`);
    
    if (overallSpeedup > 1.5) {
      console.log('  🎉 EXCELLENT: Significant performance improvements achieved!');
    } else if (overallSpeedup > 1.1) {
      console.log('  ✅ GOOD: Moderate performance improvements achieved!');
    } else {
      console.log('  ⚠️  MINIMAL: Performance improvements are minimal');
    }
  }
  
  console.log('\n📊 DETAILED RESULTS:');
  console.log(JSON.stringify(results, null, 2));
}