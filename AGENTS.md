# Root AGENTS.md – Guidance for Codex

## Project summary
- **oblix** is a self‑contained browser-based neural network playground. It lets users build, train and visualize models entirely in JavaScript.
- `index.html` serves as the entry page and loads ES modules from the `src` directory, starting with `src/main.js`. No build or installation steps are required—just open the HTML file in a browser.

## Development notes
- Keep the project fully self-contained in plain JavaScript (no build system or external dependencies).
- Use 2-space indentation and semicolons, matching the style in the existing modules within `src`. ​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=20 path=src/main.js git_url="https://github.com/bfuzzy1/oblix/blob/main/src/main.js#L1-L20"}​
- There are currently tests in `tests/`.
- You must run tests AFTER each code change you make. Use `node tests/run.js`.
- When adding features, ensure the UI remains simple and educational.
- Documentation should be thorough and complete.

## Pull requests
You must use the below format for your pr's.

**Context**
Gives the reviewer some context about the work and why this change is being made, the WHY you are doing this. This field goes more into the project perspective.

**Description**
Provide a detailed description of how exactly this task will be accomplished. This can be something technical. What specific steps will be taken to achieve the goal? This should include details on service integration, job logic, implementation, etc.

**Changes in the codebase**
This is where becomes technical. Here is where you can be more focused on the engineering side of your solution. Include information about the functionality you are adding or modifying, as well as any refactoring or improvement of existing code.

## How to run
1. Open `index.html` in a modern web browser.
   It imports `src/main.js`, which loads the rest of the modules from the `src` directory. No additional steps are required.
