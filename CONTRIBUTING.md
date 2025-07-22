# Contributing to oblix

Thank you for your interest in contributing to oblix! This document provides comprehensive guidelines for contributing to the project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Feature Development](#feature-development)
- [Bug Reports](#bug-reports)
- [Documentation](#documentation)

## Getting Started

### Prerequisites
- **Node.js** (version 14 or higher)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **Git** for version control
- **Text editor** with JavaScript support

### Fork and Clone
1. **Fork** the oblix repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/oblix.git
   cd oblix
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/bfuzzy1/oblix.git
   ```

## Development Setup

### Local Development
1. **No build process required** - oblix is pure JavaScript
2. **Open `index.html`** in your browser to test changes
3. **Use browser developer tools** for debugging
4. **Run tests** after each change:
   ```bash
   node tests/run.js
   ```

### Testing Your Changes
1. **Open `index.html`** in a modern browser
2. **Test all major features:**
   - Network construction
   - Data generation
   - Training process
   - Model save/load
   - Visualizations
3. **Test across different browsers** if possible
4. **Verify performance** with benchmarks:
   ```bash
   node benchmark/run.js
   ```

## Code Style

### JavaScript Standards
- **ES6+ Features:** Use modern JavaScript features
- **2-space indentation:** No tabs
- **Semicolons required:** At end of statements
- **camelCase:** For variables and functions
- **PascalCase:** For classes
- **ES6 Modules:** Use import/export syntax

### Code Organization
```javascript
// File structure example
import { dependency } from './dependency.js';

// Constants
const CONSTANT_VALUE = 42;

// Main function/class
export function mainFunction() {
  // Implementation
}

// Helper functions
function helperFunction() {
  // Helper implementation
}
```

### Comments and Documentation
```javascript
/**
 * Performs forward pass through the neural network
 * @param {Float32Array} input - Input tensor
 * @param {object} options - Forward pass options
 * @returns {Float32Array} Output tensor
 */
function forwardPass(input, options = {}) {
  // Implementation with inline comments for complex logic
}
```

### Error Handling
```javascript
function robustFunction(input) {
  // Input validation
  if (!input || !(input instanceof Float32Array)) {
    throw new Error('Input must be a Float32Array');
  }
  
  try {
    // Implementation
    return result;
  } catch (error) {
    console.error('Function failed:', error.message);
    throw error;
  }
}
```

## Testing

### Running Tests
```bash
# Run all tests
node tests/run.js

# Run specific test file
node tests/test_network.js
```

### Writing Tests
```javascript
// Example test structure
import { testFunction } from '../src/utils.js';

// Test basic functionality
console.log('Testing basic functionality...');
const result = testFunction(input);
if (result !== expected) {
  console.error('Test failed: expected', expected, 'got', result);
  process.exit(1);
}

// Test edge cases
console.log('Testing edge cases...');
try {
  testFunction(null);
  console.error('Should have thrown error for null input');
  process.exit(1);
} catch (error) {
  // Expected error
}

console.log('All tests passed!');
```

### Test Categories
- **Unit Tests:** Individual function testing
- **Integration Tests:** Layer interaction testing
- **Performance Tests:** Benchmark critical functions
- **UI Tests:** Interface functionality verification

### Test Guidelines
- **Test all new functions** thoroughly
- **Include edge cases** and error conditions
- **Test performance** for critical functions
- **Verify browser compatibility** for UI changes
- **Update tests** when modifying existing functionality

## Pull Request Process

### Before Submitting
1. **Run all tests:** `node tests/run.js`
2. **Run benchmarks:** `node benchmark/run.js`
3. **Test in browser:** Open `index.html` and verify functionality
4. **Update documentation:** If adding new features
5. **Check code style:** Follow project conventions

### Pull Request Format
Use the following template for all pull requests:

#### Context
Explain the motivation and project perspective:
- Why is this change needed?
- How does it fit into the project goals?
- What problem does it solve?

#### Description
Provide detailed technical implementation:
- Specific steps taken to implement the feature
- Technical decisions and their rationale
- Integration points with existing code
- Performance implications

#### Changes
Detail specific code modifications:
- Files modified and why
- New functionality added
- Breaking changes (if any)
- Testing approach

### Review Process
1. **Automated checks:** Tests and benchmarks must pass
2. **Code review:** Maintainers will review your code
3. **Functionality testing:** Verify features work as intended
4. **Documentation review:** Ensure documentation is updated
5. **Performance review:** Check for performance regressions

## Feature Development

### Adding New Layer Types
1. **Implement forward/backward methods** in `src/layers.js`
2. **Update network configuration** in `src/network.js`
3. **Add UI support** in `src/main.js`
4. **Create comprehensive tests** in `tests/`
5. **Update documentation** in README.md and API.md

### Adding New Activation Functions
1. **Add function** to `src/activations.js`
2. **Include derivative** function
3. **Add to UI** activation function dropdown
4. **Create tests** for forward/backward pass
5. **Add benchmark** for performance measurement

### Adding New Optimizers
1. **Implement optimizer** in `src/optimizers.js`
2. **Handle state management** for optimizer-specific variables
3. **Add to UI** optimizer selection
4. **Test convergence** on simple problems
5. **Update documentation** with optimizer details

### Adding New Data Generators
1. **Implement generator** in `src/utils.js`
2. **Add to UI** data pattern dropdown
3. **Handle parameters** for data-specific options
4. **Create tests** for data generation
5. **Add benchmark** for performance measurement

## Bug Reports

### Reporting Bugs
When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Browser and version** information
5. **Console errors** or stack traces
6. **Minimal test case** if possible

### Bug Report Template
```
**Bug Description:**
[Clear description of the issue]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Environment:**
- Browser: [Chrome/Firefox/Safari/Edge]
- Version: [Browser version]
- OS: [Operating system]

**Additional Information:**
[Any other relevant details]
```

## Documentation

### Documentation Standards
- **Keep documentation up-to-date** with code changes
- **Use clear, concise language**
- **Include practical examples**
- **Provide step-by-step instructions**
- **Update API documentation** for new functions

### Documentation Files
- **README.md:** Project overview and quick start
- **API.md:** Comprehensive API documentation
- **AGENTS.md:** Development guidelines
- **ROADMAP.md:** Future development plans
- **CONTRIBUTING.md:** This file

### Updating Documentation
1. **Update relevant files** when adding features
2. **Include code examples** for new functionality
3. **Update API documentation** for new functions
4. **Add troubleshooting information** for common issues
5. **Keep installation instructions** current

## Code Review Guidelines

### Review Checklist
- [ ] **Functionality:** Feature works as intended
- [ ] **Testing:** All tests pass
- [ ] **Performance:** No significant regressions
- [ ] **Documentation:** Updated where needed
- [ ] **UI Consistency:** Integrates well with existing interface
- [ ] **Error Handling:** Proper error messages
- [ ] **Browser Compatibility:** Works in target browsers
- [ ] **Code Style:** Follows project conventions

### Review Process
1. **Automated checks** must pass
2. **Manual testing** of new features
3. **Code review** by maintainers
4. **Performance verification** if applicable
5. **Documentation review** for completeness

## Performance Considerations

### Optimization Guidelines
- **Use Float32Array** for numerical computations
- **Minimize memory allocations** in hot paths
- **Profile performance** for critical functions
- **Add benchmarks** for new features
- **Test performance** across different browsers

### Benchmarking
```bash
# Run performance benchmarks
node benchmark/run.js

# Compare before/after changes
# Ensure no significant performance regressions
```

## Security Considerations

### Input Validation
- **Validate all user inputs** before processing
- **Sanitize file uploads** and data imports
- **Handle malformed data** gracefully
- **Use secure random generation** for dropout masks

### File Operations
- **Validate file types** and sizes
- **Sanitize file names** and paths
- **Handle file operation errors** properly
- **Use secure file handling** practices

## Community Guidelines

### Communication
- **Be respectful** and constructive
- **Ask questions** when unsure
- **Provide context** for suggestions
- **Help others** when possible

### Contribution Recognition
- **All contributors** are acknowledged
- **Significant contributions** are highlighted
- **Documentation updates** are valued
- **Testing contributions** are appreciated

## Getting Help

### Resources
- **GitHub Issues:** For bug reports and feature requests
- **GitHub Discussions:** For questions and discussions
- **Documentation:** README.md, API.md, and other docs
- **Code Examples:** Test files and examples

### Contact
- **Open an issue** for bug reports
- **Start a discussion** for questions
- **Submit a pull request** for contributions
- **Review existing issues** before creating new ones

Thank you for contributing to oblix! Your contributions help make this project better for everyone.