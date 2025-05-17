# Root AGENTS.md – Guidance for Codex

## Project summary
- **oblix** is a self‑contained browser-based neural network playground.  
  It lets users build, train and visualize models entirely in JavaScript. ​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=5 path=README.md git_url="https://github.com/bfuzzy1/oblix/blob/main/README.md#L1-L5"}​
- The single HTML file (`index.html`) loads all logic from `src/oblix.js`. Simply open it in a browser – no build or installation steps are required. ​:codex-file-citation[codex-file-citation]{line_range_start=25 line_range_end=33 path=README.md git_url="https://github.com/bfuzzy1/oblix/blob/main/README.md#L25-L33"}​
- The 2025 roadmap emphasizes educational features such as pause/resume training and a visual backpropagation step-through. ​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=24 path=ROADMAP.md git_url="https://github.com/bfuzzy1/oblix/blob/main/ROADMAP.md#L1-L24"}​

## Development notes
- Keep the project fully self-contained in plain JavaScript (no build system or external dependencies).
- Use 2-space indentation and semicolons, matching the style in `src/oblix.js`. ​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=20 path=src/oblix.js git_url="https://github.com/bfuzzy1/oblix/blob/main/src/oblix.js#L1-L20"}​
- There are currently no automated tests or linters. Manual validation is sufficient.
- When adding features, ensure the UI remains simple and educational, in line with the roadmap.

## How to run
1. Open `index.html` in a modern web browser.  
   This loads `src/oblix.js` and provides the interactive UI. No additional steps are required.
