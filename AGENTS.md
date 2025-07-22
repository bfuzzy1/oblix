# AGENTS.md – Development Guidelines for oblix

## Project Overview

**oblix** is a self-contained browser-based neural network playground written entirely in pure JavaScript. It provides an interactive environment for building, training, and visualizing neural networks directly in the browser without any external dependencies.

### Core Architecture
- **Entry Point:** `index.html` serves as the main application page
- **Module System:** ES6 modules loaded from the `src/` directory
- **Main Controller:** `src/main.js` orchestrates the UI and neural network interactions
- **Neural Network Core:** `src/network.js` contains the main `Oblix` class
- **Layer Operations:** `src/layers.js` implements attention, dropout, normalization, and softmax
- **Activation Functions:** `src/activations.js` provides various activation functions
- **Optimizers:** `src/optimizers.js` implements SGD, Adam, RMSprop, and AdamW
- **Utilities:** `src/utils.js` contains helper functions for data generation and metrics

## Development Standards

### Code Style & Conventions
- **Indentation:** 2 spaces (no tabs)
- **Semicolons:** Required at end of statements
- **Naming:** camelCase for variables and functions, PascalCase for classes
- **Modules:** ES6 import/export syntax
- **Comments:** JSDoc style for functions, inline comments for complex logic
- **Error Handling:** Comprehensive try-catch blocks with meaningful error messages

### JavaScript Standards
- **ES6+ Features:** Arrow functions, destructuring, template literals, modules
- **Typed Arrays:** Float32Array for all numerical computations
- **Async/Await:** For asynchronous operations (file I/O, training)
- **Strict Mode:** Implicit in ES6 modules

### Testing Requirements
- **Test Coverage:** Every new feature must have corresponding tests
- **Test Location:** All tests in `tests/` directory
- **Test Runner:** `node tests/run.js` must pass before any commit
- **Test Types:**
  - Unit tests for individual functions
  - Integration tests for layer operations
  - Performance benchmarks for critical functions

### Performance Considerations
- **Memory Efficiency:** Use Float32Array for all numerical data
- **Computation:** Optimize mathematical operations where possible
- **Random Generation:** Use `crypto.getRandomValues` for dropout masks
- **Visualization:** Efficient canvas rendering for real-time updates

## Feature Development Guidelines

### Adding New Layer Types
1. **Implementation:** Add forward/backward methods in `src/layers.js`
2. **Integration:** Update `src/network.js` layer configuration
3. **UI Support:** Add layer type to main.js UI generation
4. **Testing:** Create comprehensive tests in `tests/`
5. **Documentation:** Update README.md with new layer capabilities

### Adding New Activation Functions
1. **Implementation:** Add function to `src/activations.js`
2. **Derivative:** Include corresponding derivative function
3. **UI Integration:** Add to activation function dropdown
4. **Testing:** Verify forward/backward pass correctness
5. **Benchmarking:** Add performance benchmark

### Adding New Optimizers
1. **Implementation:** Add to `src/optimizers.js`
2. **State Management:** Handle optimizer-specific state variables
3. **UI Integration:** Add to optimizer selection dropdown
4. **Testing:** Verify convergence on simple problems
5. **Documentation:** Update optimizer descriptions

### Adding New Data Generators
1. **Implementation:** Add to `src/utils.js`
2. **UI Integration:** Add to data pattern dropdown
3. **Parameter Handling:** Update UI for data-specific parameters
4. **Testing:** Verify data generation correctness
5. **Benchmarking:** Add performance benchmark

## Quality Assurance

### Code Review Checklist
- [ ] **Functionality:** Feature works as intended
- [ ] **Testing:** All tests pass (`node tests/run.js`)
- [ ] **Performance:** No significant performance regressions
- [ ] **Documentation:** README.md updated if needed
- [ ] **UI Consistency:** New features integrate well with existing UI
- [ ] **Error Handling:** Proper error messages and graceful failures
- [ ] **Browser Compatibility:** Works in modern browsers

### Testing Strategy
- **Unit Tests:** Individual function testing
- **Integration Tests:** Layer interaction testing
- **End-to-End Tests:** Complete training workflow testing
- **Performance Tests:** Benchmark critical functions
- **UI Tests:** Interface functionality verification

### Performance Benchmarks
- **Activation Functions:** Measure speed of mathematical operations
- **Layer Operations:** Test forward/backward pass efficiency
- **Data Generation:** Verify synthetic data creation speed
- **Training:** Monitor end-to-end training performance

## Pull Request Process

### Required Format
**Context:** Explain the motivation and project perspective
- Why is this change needed?
- How does it fit into the project goals?
- What problem does it solve?

**Description:** Detailed technical implementation
- Specific steps taken to implement the feature
- Technical decisions and their rationale
- Integration points with existing code

**Changes:** Specific code modifications
- Files modified and why
- New functionality added
- Breaking changes (if any)
- Performance implications

### Review Criteria
- **Code Quality:** Clean, readable, maintainable code
- **Test Coverage:** Comprehensive testing of new functionality
- **Documentation:** Updated documentation where needed
- **Performance:** No significant performance regressions
- **UI/UX:** Consistent with existing interface design

## Maintenance Guidelines

### Bug Fixes
1. **Reproduce:** Create minimal test case demonstrating the bug
2. **Fix:** Implement the fix with proper error handling
3. **Test:** Verify fix resolves the issue
4. **Document:** Update relevant documentation if needed

### Performance Improvements
1. **Profile:** Identify performance bottlenecks
2. **Optimize:** Implement targeted optimizations
3. **Benchmark:** Measure improvement with benchmarks
4. **Test:** Ensure optimizations don't break functionality

### Documentation Updates
1. **Accuracy:** Ensure all documentation reflects current implementation
2. **Completeness:** Cover all features and capabilities
3. **Clarity:** Make documentation accessible to target audience
4. **Examples:** Include practical usage examples

## Technical Debt Management

### Code Refactoring
- **Identify:** Areas needing improvement
- **Plan:** Systematic refactoring approach
- **Execute:** Implement changes incrementally
- **Test:** Ensure refactoring maintains functionality
- **Document:** Update documentation to reflect changes

### Dependency Management
- **Minimal Dependencies:** Keep external dependencies to absolute minimum
- **Self-Contained:** Core neural network logic should be pure JavaScript
- **Browser APIs:** Use standard web APIs where possible
- **Fallbacks:** Provide graceful degradation for unsupported features

## Security Considerations

### Input Validation
- **Sanitize:** All user inputs before processing
- **Validate:** Data types and ranges
- **Handle:** Malformed inputs gracefully

### File Operations
- **Validate:** File types and sizes
- **Sanitize:** File names and paths
- **Error Handle:** File operation failures

### Random Number Generation
- **Secure:** Use `crypto.getRandomValues` for cryptographic operations
- **Deterministic:** Provide fallback for testing scenarios
- **Performance:** Balance security with performance requirements

## Deployment Guidelines

### GitHub Pages Deployment
- **Static Files:** Ensure all files are properly structured
- **Module Loading:** Verify ES6 modules work in production
- **Cross-Origin:** Handle CORS issues if they arise
- **Performance:** Optimize for web deployment

### Local Development
- **Setup:** Simple file-based development
- **Testing:** Local test execution
- **Debugging:** Browser developer tools integration
- **Hot Reload:** Manual refresh for development

## Future Considerations

### Scalability
- **Web Workers:** Consider for heavy computations
- **WebGPU:** Future acceleration possibilities
- **Modular Architecture:** Maintain extensible design
- **Plugin System:** Community contribution framework

### Educational Focus
- **Simplicity:** Keep UI intuitive for learning
- **Visualization:** Enhance educational value of visualizations
- **Documentation:** Comprehensive tutorials and examples
- **Community:** Foster learning and experimentation

This document serves as the primary reference for all development work on oblix. All contributors should familiarize themselves with these guidelines and follow them consistently.
