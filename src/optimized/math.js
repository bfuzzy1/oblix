// Optimized mathematical operations for neural networks
export const optimizedMath = {
  // Pre-allocated buffers to reduce memory allocations
  tempBuffers: {
    small: new Float32Array(1024),
    medium: new Float32Array(4096),
    large: new Float32Array(16384)
  },

  // Get appropriate buffer size to minimize allocations
  getBuffer(size) {
    if (size <= 1024) return this.tempBuffers.small;
    if (size <= 4096) return this.tempBuffers.medium;
    if (size <= 16384) return this.tempBuffers.large;
    return new Float32Array(size);
  },

  // Optimized matrix multiplication with better cache locality
  matrixMultiply(a, b, aRows, aCols, bCols, result = null) {
    if (!result) {
      result = new Float32Array(aRows * bCols);
    }

    // Use better cache locality by accessing memory sequentially
    for (let i = 0; i < aRows; i++) {
      const aOffset = i * aCols;
      const rOffset = i * bCols;
      
      for (let j = 0; j < bCols; j++) {
        let sum = 0;
        
        // Unroll by 4 for better performance
        const unrollSize = 4;
        const remainder = aCols % unrollSize;
        
        for (let k = 0; k < aCols - remainder; k += unrollSize) {
          sum += a[aOffset + k] * b[k * bCols + j];
          sum += a[aOffset + k + 1] * b[(k + 1) * bCols + j];
          sum += a[aOffset + k + 2] * b[(k + 2) * bCols + j];
          sum += a[aOffset + k + 3] * b[(k + 3) * bCols + j];
        }
        
        // Handle remainder
        for (let k = aCols - remainder; k < aCols; k++) {
          sum += a[aOffset + k] * b[k * bCols + j];
        }
        
        result[rOffset + j] = sum;
      }
    }
    
    return result;
  },

  // Optimized activation functions with minimal overhead
  fastActivation(x, type) {
    // Direct computation without lookup tables for better performance
    switch (type) {
    case 'tanh': return Math.tanh(x);
    case 'sigmoid': return 1 / (1 + Math.exp(-x));
    case 'relu': return x > 0 ? x : 0;
    case 'leakyrelu': return x > 0 ? x : 0.01 * x;
    case 'gelu': {
      const k = 0.7978845608;
      const x3 = x * x * x;
      return 0.5 * x * (1 + Math.tanh(k * (x + 0.044715 * x3)));
    }
    default: return x;
    }
  },

  // Vectorized operations with minimal allocations
  vectorAdd(a, b, result = null) {
    const len = a.length;
    if (!result) result = new Float32Array(len);
    
    // Unroll by 8 for better performance
    const unrollSize = 8;
    const remainder = len % unrollSize;
    
    for (let i = 0; i < len - remainder; i += unrollSize) {
      result[i] = a[i] + b[i];
      result[i + 1] = a[i + 1] + b[i + 1];
      result[i + 2] = a[i + 2] + b[i + 2];
      result[i + 3] = a[i + 3] + b[i + 3];
      result[i + 4] = a[i + 4] + b[i + 4];
      result[i + 5] = a[i + 5] + b[i + 5];
      result[i + 6] = a[i + 6] + b[i + 6];
      result[i + 7] = a[i + 7] + b[i + 7];
    }
    
    for (let i = len - remainder; i < len; i++) {
      result[i] = a[i] + b[i];
    }
    
    return result;
  },

  vectorMultiply(a, b, result = null) {
    const len = a.length;
    if (!result) result = new Float32Array(len);
    
    // Unroll by 8 for better performance
    const unrollSize = 8;
    const remainder = len % unrollSize;
    
    for (let i = 0; i < len - remainder; i += unrollSize) {
      result[i] = a[i] * b[i];
      result[i + 1] = a[i + 1] * b[i + 1];
      result[i + 2] = a[i + 2] * b[i + 2];
      result[i + 3] = a[i + 3] * b[i + 3];
      result[i + 4] = a[i + 4] * b[i + 4];
      result[i + 5] = a[i + 5] * b[i + 5];
      result[i + 6] = a[i + 6] * b[i + 6];
      result[i + 7] = a[i + 7] * b[i + 7];
    }
    
    for (let i = len - remainder; i < len; i++) {
      result[i] = a[i] * b[i];
    }
    
    return result;
  },

  vectorScale(a, scalar, result = null) {
    const len = a.length;
    if (!result) result = new Float32Array(len);
    
    // Unroll by 8 for better performance
    const unrollSize = 8;
    const remainder = len % unrollSize;
    
    for (let i = 0; i < len - remainder; i += unrollSize) {
      result[i] = a[i] * scalar;
      result[i + 1] = a[i + 1] * scalar;
      result[i + 2] = a[i + 2] * scalar;
      result[i + 3] = a[i + 3] * scalar;
      result[i + 4] = a[i + 4] * scalar;
      result[i + 5] = a[i + 5] * scalar;
      result[i + 6] = a[i + 6] * scalar;
      result[i + 7] = a[i + 7] * scalar;
    }
    
    for (let i = len - remainder; i < len; i++) {
      result[i] = a[i] * scalar;
    }
    
    return result;
  },

  // Optimized softmax with numerical stability
  softmax(input, result = null) {
    const len = input.length;
    if (!result) result = new Float32Array(len);
    
    // Find max for numerical stability
    let max = -Infinity;
    for (let i = 0; i < len; i++) {
      if (input[i] > max) max = input[i];
    }
    
    // Compute exponentials
    let sum = 0;
    for (let i = 0; i < len; i++) {
      result[i] = Math.exp(input[i] - max);
      sum += result[i];
    }
    
    // Normalize
    const invSum = 1 / sum;
    for (let i = 0; i < len; i++) {
      result[i] *= invSum;
    }
    
    return result;
  },

  // Memory-efficient gradient computation
  computeGradients(weights, gradients, learningRate, result = null) {
    const len = weights.length;
    if (!result) result = new Float32Array(len);
    
    for (let i = 0; i < len; i++) {
      result[i] = weights[i] - learningRate * gradients[i];
    }
    
    return result;
  },

  // Batch operations for better cache locality
  batchActivation(inputs, activationType, result = null) {
    const len = inputs.length;
    if (!result) result = new Float32Array(len);
    
    // Use direct computation for better performance
    for (let i = 0; i < len; i++) {
      result[i] = this.fastActivation(inputs[i], activationType);
    }
    
    return result;
  },

  // Optimized matrix-vector multiplication
  matrixVectorMultiply(matrix, vector, rows, cols, result = null) {
    if (!result) result = new Float32Array(rows);
    
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const mOffset = i * cols;
      
      // Unroll by 4
      const unrollSize = 4;
      const remainder = cols % unrollSize;
      
      for (let j = 0; j < cols - remainder; j += unrollSize) {
        sum += matrix[mOffset + j] * vector[j];
        sum += matrix[mOffset + j + 1] * vector[j + 1];
        sum += matrix[mOffset + j + 2] * vector[j + 2];
        sum += matrix[mOffset + j + 3] * vector[j + 3];
      }
      
      for (let j = cols - remainder; j < cols; j++) {
        sum += matrix[mOffset + j] * vector[j];
      }
      
      result[i] = sum;
    }
    
    return result;
  }
};