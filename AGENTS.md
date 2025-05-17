# Root AGENTS.md – Guidance for Codex

## Project summary
- **oblix** is a self‑contained browser-based neural network playground. It lets users build, train and visualize models entirely in JavaScript.
- `index.html` serves as the entry page and loads ES modules from the `src` directory, starting with `src/main.js`. No build or installation steps are required—just open the HTML file in a browser.
- The 2025 roadmap emphasizes educational features such as pause/resume training and a visual backpropagation step-through.

## Development notes
- Keep the project fully self-contained in plain JavaScript (no build system or external dependencies).
- Use 2-space indentation and semicolons, matching the style in the existing modules within `src`. ​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=20 path=src/main.js git_url="https://github.com/bfuzzy1/oblix/blob/main/src/main.js#L1-L20"}​
- There are currently no automated tests or linters. Manual validation is sufficient.
- When adding features, ensure the UI remains simple and educational, in line with the roadmap.

## How to run
1. Open `index.html` in a modern web browser.
   It imports `src/main.js`, which loads the rest of the modules from the `src` directory. No additional steps are required.
