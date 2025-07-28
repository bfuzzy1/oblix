#!/usr/bin/env node

/**
 * Daily Kaizen Script
 * 
 * Following Japanese-Level Code Quality Rule-Book principles:
 * - Monozukuri (craftsmanship): Treat every line as a product meant to last decades
 * - Kaizen (continuous improvement): 1% better every day
 * - Sustainable pace: Work rhythm that can be maintained for 20 years
 * - Wabi-sabi (graceful imperfection): Ship the simplest code that solves today's need
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const KAIZEN_LOG_PATH = '../KAIZEN_LOG.md';

/**
 * Runs a command and returns the result
 * @param {string} command - Command to execute
 * @returns {string} Command output
 */
function runCommand(command) {
  try {
    return execSync(command, { encoding: 'utf8', stdio: 'pipe' });
  } catch (error) {
    return error.stdout || error.message;
  }
}

/**
 * Checks if all quality gates pass
 * @returns {Object} Quality check results
 */
function runQualityGates() {
  console.log('🎯 Running Quality Gates...\n');
  
  const results = {
    lint: false,
    format: false,
    tests: false,
    overall: false
  };

  // Check linting
  console.log('📋 Checking code style...');
  try {
    const lintResult = runCommand('npm run lint');
    if (lintResult.includes('problems') && !lintResult.includes('0 problems')) {
      console.log('❌ Linting issues found. Run "npm run lint:fix" to fix.');
      console.log(lintResult);
    } else {
      console.log('✅ Code style looks good!');
      results.lint = true;
    }
  } catch (error) {
    console.log('❌ Linting failed:', error.message);
  }

  // Check formatting
  console.log('\n🎨 Checking code formatting...');
  try {
    const formatResult = runCommand('npm run format:check');
    if (formatResult.includes('warnings')) {
      console.log('❌ Formatting issues found. Run "npm run format" to fix.');
    } else {
      console.log('✅ Code formatting looks good!');
      results.format = true;
    }
  } catch (error) {
    console.log('❌ Format check failed:', error.message);
  }

  // Check tests
  console.log('\n🧪 Running tests...');
  try {
    const testResult = runCommand('npm test');
    if (testResult.includes('FAIL')) {
      console.log('❌ Tests failed!');
      console.log(testResult);
    } else {
      console.log('✅ All tests passing!');
      results.tests = true;
    }
  } catch (error) {
    console.log('❌ Tests failed:', error.message);
  }

  results.overall = results.lint && results.format && results.tests;
  return results;
}

/**
 * Suggests kaizen improvements based on current state
 * @returns {Array<string>} List of suggested improvements
 */
function suggestKaizenImprovements() {
  const suggestions = [
    '🔧 Pick one function to refactor (aim for ≤ 50 lines)',
    '📝 Add JSDoc comments to undocumented functions',
    '🧪 Write a test for an edge case',
    '🎨 Improve a variable name for clarity',
    '📚 Update documentation for a complex business rule',
    '⚡ Extract a tiny helper function',
    '🛡️ Add error handling to a function',
    '📊 Improve a console message for debugging'
  ];
  
  return suggestions;
}

/**
 * Logs today's kaizen achievement
 * @param {string} achievement - What was accomplished today
 */
function logKaizenAchievement(achievement) {
  const today = new Date().toISOString().split('T')[0];
  const logEntry = `\n### ${today} - Daily Kaizen\n**Target:** ${achievement}\n**Achievement:** ✅ Complete\n- ${achievement}\n\n**Next Target:** Choose from suggestions below\n`;
  
  try {
    const logPath = path.resolve(KAIZEN_LOG_PATH);
    if (fs.existsSync(logPath)) {
      const content = fs.readFileSync(logPath, 'utf8');
      const updatedContent = content.replace(
        '## Daily Kaizen Tracking',
        `## Daily Kaizen Tracking${logEntry}`
      );
      fs.writeFileSync(logPath, updatedContent);
      console.log('📝 Kaizen achievement logged!');
    }
  } catch (error) {
    console.log('⚠️ Could not update kaizen log:', error.message);
  }
}

/**
 * Main kaizen process
 */
function main() {
  console.log('🎯 Daily Kaizen - Japanese-Level Code Quality\n');
  console.log('"The fastest way to go fast is to never slow down."\n');
  
  // Run quality gates
  const qualityResults = runQualityGates();
  
  console.log('\n📊 Quality Gates Summary:');
  console.log(`Linting: ${qualityResults.lint ? '✅' : '❌'}`);
  console.log(`Formatting: ${qualityResults.format ? '✅' : '❌'}`);
  console.log(`Tests: ${qualityResults.tests ? '✅' : '❌'}`);
  console.log(`Overall: ${qualityResults.overall ? '✅' : '❌'}`);
  
  if (qualityResults.overall) {
    console.log('\n🎉 All quality gates passed! Time for kaizen improvement.');
  } else {
    console.log('\n🔧 Fix quality issues first, then continue with kaizen.');
  }
  
  // Suggest improvements
  console.log('\n💡 Today\'s Kaizen Suggestions:');
  const suggestions = suggestKaizenImprovements();
  suggestions.forEach((suggestion, index) => {
    console.log(`${index + 1}. ${suggestion}`);
  });
  
  console.log('\n🎯 Pick ONE improvement for today:');
  console.log('   - Keep it small (1% better)');
  console.log('   - Focus on craftsmanship over cleverness');
  console.log('   - Make it maintainable for 20 years');
  
  // Interactive prompt for achievement logging
  console.log('\n📝 What kaizen improvement did you complete today?');
  console.log('(Press Enter to skip logging)');
  
  // In a real implementation, you'd use readline for input
  // For now, we'll just show the structure
  console.log('\n💡 To log your achievement, edit KAIZEN_LOG.md');
  
  console.log('\n🏆 Remember: 1% better every day compounds into excellence!');
}

// Run the kaizen process
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { runQualityGates, suggestKaizenImprovements, logKaizenAchievement };