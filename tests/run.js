import fs from 'fs';
import path from 'path';
import assert from 'assert';

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const files = fs.readdirSync(__dirname).filter(f => f.endsWith('.js') && f !== 'run.js');

let passed = 0;
let failed = 0;

for (const file of files) {
  try {
    const mod = await import(`./${file}`);
    if (typeof mod.run === 'function') {
      await mod.run(assert);
      console.log(`${file}: OK`);
      passed++;
    } else {
      console.log(`${file}: No run() exported`);
    }
  } catch (err) {
    failed++;
    console.error(`${file}: FAIL`);
    console.error(err);
  }
}

if (failed > 0) {
  console.error(`\n${failed} test file(s) failed.`);
  process.exit(1);
} else {
  console.log(`\nAll ${passed} test file(s) passed.`);
}
