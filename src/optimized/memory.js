// Memory optimization utilities for neural networks
export class MemoryOptimizer {
  constructor() {
    this.pools = new Map();
    this.maxPoolSize = 100;
    this.allocatedBuffers = new Set();
  }

  // Get a buffer from pool or create new one
  getBuffer(size, type = Float32Array) {
    const key = `${type.name}_${size}`;
    
    if (!this.pools.has(key)) {
      this.pools.set(key, []);
    }
    
    const pool = this.pools.get(key);
    
    if (pool.length > 0) {
      const buffer = pool.pop();
      this.allocatedBuffers.add(buffer);
      return buffer;
    }
    
    const buffer = new type(size);
    this.allocatedBuffers.add(buffer);
    return buffer;
  }

  // Return buffer to pool for reuse
  returnBuffer(buffer) {
    if (!this.allocatedBuffers.has(buffer)) {
      console.warn('Attempting to return unallocated buffer');
      return;
    }
    
    this.allocatedBuffers.delete(buffer);
    
    const key = `${buffer.constructor.name}_${buffer.length}`;
    const pool = this.pools.get(key);
    
    if (pool && pool.length < this.maxPoolSize) {
      // Clear buffer before returning to pool
      buffer.fill(0);
      pool.push(buffer);
    }
  }

  // Clear all pools
  clearPools() {
    this.pools.clear();
    this.allocatedBuffers.clear();
  }

  // Get memory usage statistics
  getMemoryStats() {
    let totalPooled = 0;
    let totalAllocated = 0;
    
    for (const [key, pool] of this.pools) {
      const size = parseInt(key.split('_')[1]);
      totalPooled += pool.length * size * 4; // 4 bytes per float32
    }
    
    totalAllocated = this.allocatedBuffers.size * 1024; // Approximate
    
    return {
      pooledBuffers: totalPooled,
      allocatedBuffers: totalAllocated,
      totalMemory: totalPooled + totalAllocated
    };
  }
}

// Global memory optimizer instance
export const memoryOptimizer = new MemoryOptimizer();

// Memory-efficient array operations
export const memoryOptimizedOps = {
  // Create array with minimal allocation
  createArray(size, type = Float32Array) {
    return memoryOptimizer.getBuffer(size, type);
  },

  // Return array to pool
  returnArray(array) {
    memoryOptimizer.returnBuffer(array);
  },

  // Copy array with optional reuse
  copyArray(source, target = null) {
    if (!target) {
      target = this.createArray(source.length, source.constructor);
    }
    target.set(source);
    return target;
  },

  // Zero array without allocation
  zeroArray(array) {
    array.fill(0);
  },

  // Resize array with minimal copying
  resizeArray(array, newSize, fillValue = 0) {
    if (array.length === newSize) return array;
    
    const newArray = this.createArray(newSize, array.constructor);
    
    if (newSize > array.length) {
      // Growing - copy old data and fill new space
      newArray.set(array);
      if (fillValue !== 0) {
        for (let i = array.length; i < newSize; i++) {
          newArray[i] = fillValue;
        }
      }
    } else {
      // Shrinking - copy only what fits
      newArray.set(array.subarray(0, newSize));
    }
    
    this.returnArray(array);
    return newArray;
  },

  // Batch operations with memory reuse
  batchOperation(arrays, operation, result = null) {
    if (!result) {
      result = this.createArray(arrays[0].length);
    }
    
    operation(arrays, result);
    return result;
  },

  // Memory-efficient matrix operations
  async matrixMultiply(a, b, aRows, aCols, bCols, result = null) {
    if (!result) {
      result = this.createArray(aRows * bCols);
    }
    
    // Use optimized math operations
    const { optimizedMath } = await import('./math.js');
    return optimizedMath.matrixMultiply(a, b, aRows, aCols, bCols, result);
  },

  // Memory-efficient activation
  async applyActivation(input, activationType, result = null) {
    if (!result) {
      result = this.createArray(input.length);
    }
    
    const { optimizedMath } = await import('./math.js');
    return optimizedMath.batchActivation(input, activationType, result);
  },

  // Clear all pools
  clearPools() {
    memoryOptimizer.clearPools();
  }
};

// Memory monitoring utilities
export const memoryMonitor = {
  snapshots: [],
  maxSnapshots: 10,

  takeSnapshot(label = '') {
    const snapshot = {
      timestamp: performance.now(),
      label,
      memory: memoryOptimizer.getMemoryStats(),
      heap: performance.memory ? {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit
      } : null
    };
    
    this.snapshots.push(snapshot);
    
    if (this.snapshots.length > this.maxSnapshots) {
      this.snapshots.shift();
    }
    
    return snapshot;
  },

  getMemoryTrend() {
    if (this.snapshots.length < 2) return null;
    
    const recent = this.snapshots.slice(-5);
    const trend = {
      pooledGrowth: recent[recent.length - 1].memory.pooledBuffers - recent[0].memory.pooledBuffers,
      allocatedGrowth: recent[recent.length - 1].memory.allocatedBuffers - recent[0].memory.allocatedBuffers,
      totalGrowth: recent[recent.length - 1].memory.totalMemory - recent[0].memory.totalMemory
    };
    
    return trend;
  },

  clearSnapshots() {
    this.snapshots = [];
  }
};