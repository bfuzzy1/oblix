/**
 * Helper functions following Japanese-Level Code Quality principles.
 * 
 * Kaizen principle: Extract tiny helpers & keep functions < 50 LOC
 * Wabi-sabi principle: Ship the simplest code that solves today's need
 */

/**
 * Validates that a value is a finite number.
 * Business rule: Neural network computations require finite numbers.
 * 
 * @param {*} value - Value to validate
 * @returns {boolean} True if value is a finite number
 */
export function isValidNumber(value) {
  return typeof value === 'number' && isFinite(value);
}

/**
 * Ensures an array has the expected length.
 * Business rule: Neural network layers require specific dimensions.
 * 
 * @param {Array} array - Array to validate
 * @param {number} expectedLength - Expected array length
 * @param {string} context - Context for error messages
 * @returns {boolean} True if array has expected length
 */
export function validateArrayLength(array, expectedLength, context = 'Array') {
  if (!Array.isArray(array)) {
    console.warn(`${context}: Expected array, got ${typeof array}`);
    return false;
  }
  
  if (array.length !== expectedLength) {
    console.warn(`${context}: Expected length ${expectedLength}, got ${array.length}`);
    return false;
  }
  
  return true;
}

/**
 * Safely converts input to Float32Array.
 * Business rule: Neural networks use typed arrays for performance.
 * 
 * @param {Array|Float32Array} input - Input to convert
 * @returns {Float32Array} Converted array
 */
export function toFloat32Array(input) {
  if (input instanceof Float32Array) {
    return input;
  }
  
  if (Array.isArray(input)) {
    return new Float32Array(input);
  }
  
  console.warn('Input conversion: Expected array, converting to Float32Array');
  return new Float32Array(0);
}

/**
 * Finds the index of the maximum value in an array.
 * Business rule: Classification tasks require finding the most likely class.
 * 
 * @param {Array<number>} array - Array to search
 * @returns {number} Index of maximum value (-1 if invalid)
 */
export function findMaxIndex(array) {
  if (!Array.isArray(array) || array.length === 0) {
    return -1;
  }
  
  let maxIndex = 0;
  let maxValue = array[0];
  
  for (let i = 1; i < array.length; i++) {
    if (isValidNumber(array[i]) && array[i] > maxValue) {
      maxValue = array[i];
      maxIndex = i;
    }
  }
  
  return maxIndex;
}

/**
 * Calculates the mean of an array of numbers.
 * Business rule: Statistical calculations require accurate mean computation.
 * 
 * @param {Array<number>} array - Array of numbers
 * @returns {number} Mean value (NaN if invalid)
 */
export function calculateMean(array) {
  if (!Array.isArray(array) || array.length === 0) {
    return NaN;
  }
  
  const validNumbers = array.filter(isValidNumber);
  if (validNumbers.length === 0) {
    return NaN;
  }
  
  const sum = validNumbers.reduce((acc, val) => acc + val, 0);
  return sum / validNumbers.length;
}

/**
 * Formats a number to a specified number of decimal places.
 * Business rule: Display values should be consistently formatted.
 * 
 * @param {number} value - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number string
 */
export function formatNumber(value, decimals = 3) {
  if (!isValidNumber(value)) {
    return 'NaN';
  }
  
  return value.toFixed(decimals);
}