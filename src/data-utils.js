/**
 * Data utilities for the Oblix neural network playground.
 * 
 * Following Japanese-Level Code Quality principles:
 * - Monozukuri: Each function has a single, clear responsibility
 * - Kaizen: Extracted from the massive main.js function
 * - Wabi-sabi: Simple, focused functions that solve today's need
 */

/**
 * Generates random data for neural network training.
 * Business rule: Training data should be configurable and reproducible.
 * 
 * @param {number} numSamples - Number of data samples to generate
 * @param {number} numInputs - Number of input dimensions
 * @param {number} numOutputs - Number of output dimensions (default: 1)
 * @param {number} noiseLevel - Amount of noise to add (default: 0.05)
 * @returns {string} CSV-formatted data string
 */
export function generateRandomData(numSamples, numInputs, numOutputs = 1, noiseLevel = 0.05) {
  const data = [];
  
  for (let i = 0; i < numSamples; i++) {
    const row = [];
    
    // Generate inputs
    for (let j = 0; j < numInputs; j++) {
      const value = 0.01 + Math.random() * 0.98; // Avoid exact 0 or 1
      row.push(value);
    }
    
    // Generate outputs (simple random values for now)
    for (let j = 0; j < numOutputs; j++) {
      const value = 0.01 + Math.random() * 0.98;
      row.push(value);
    }
    
    data.push(row);
  }
  
  return formatGeneratedDataToCSV(data);
}

/**
 * Formats generated data array to CSV string.
 * Business rule: Data should be consistently formatted for easy import/export.
 * 
 * @param {Array<Array<number>>} dataArray - Array of data rows
 * @returns {string} CSV-formatted string
 */
export function formatGeneratedDataToCSV(dataArray) {
  return dataArray
    .map(row => row.map(val => val.toFixed(3)).join(', '))
    .join('\n');
}

/**
 * Parses CSV string into training data format.
 * Business rule: CSV parsing should handle various input formats gracefully.
 * 
 * @param {string} csvString - CSV string to parse
 * @param {number} numOutputs - Number of output columns (default: 1)
 * @returns {Array<Object>} Array of {input, output} objects
 */
export function parseCSV(csvString, numOutputs = 1) {
  const lines = csvString.trim().split('\n');
  const data = [];
  
  for (const line of lines) {
    if (!line.trim()) continue;
    
    const values = line.split(',').map(v => parseFloat(v.trim()));
    if (values.length < numOutputs + 1) continue;
    
    const input = values.slice(0, -numOutputs);
    const output = values.slice(-numOutputs);
    
    data.push({ input, output });
  }
  
  return data;
}

/**
 * Generates XOR pattern data.
 * Business rule: XOR is a classic non-linear problem for testing neural networks.
 * 
 * @param {number} numSamples - Number of samples to generate
 * @returns {string} CSV-formatted XOR data
 */
export function generateXORData(numSamples = 100) {
  const data = [];
  
  for (let i = 0; i < numSamples; i++) {
    const a = Math.random() < 0.5 ? 0 : 1;
    const b = Math.random() < 0.5 ? 0 : 1;
    const result = (a + b) % 2; // XOR operation
    
    data.push([a, b, result]);
  }
  
  return formatGeneratedDataToCSV(data);
}

/**
 * Generates linear relationship data.
 * Business rule: Linear data helps test basic neural network capabilities.
 * 
 * @param {number} numSamples - Number of samples to generate
 * @returns {string} CSV-formatted linear data
 */
export function generateLinearData(numSamples = 100) {
  const data = [];
  const slope = 2;
  const intercept = 1;
  
  for (let i = 0; i < numSamples; i++) {
    const x = Math.random() * 10;
    const y = slope * x + intercept + (Math.random() - 0.5) * 0.5; // Add noise
    
    data.push([x, y]);
  }
  
  return formatGeneratedDataToCSV(data);
}

/**
 * Validates training data format.
 * Business rule: Invalid data should be detected early to prevent training errors.
 * 
 * @param {Array<Object>} data - Training data to validate
 * @returns {boolean} True if data is valid
 */
export function validateTrainingData(data) {
  if (!Array.isArray(data) || data.length === 0) {
    console.warn('Training data validation: Empty or invalid data array');
    return false;
  }
  
  for (const item of data) {
    if (!item.input || !item.output) {
      console.warn('Training data validation: Missing input or output');
      return false;
    }
    
    if (!Array.isArray(item.input) || !Array.isArray(item.output)) {
      console.warn('Training data validation: Input/output must be arrays');
      return false;
    }
  }
  
  return true;
}