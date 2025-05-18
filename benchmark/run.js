import fs from 'fs';
import path from 'path';

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const files = fs
  .readdirSync(__dirname)
  .filter(f => f.endsWith('.js') && f !== 'run.js')
  .sort();

for (const file of files) {
  try {
    const mod = await import(`./${file}`);
    if (typeof mod.run === 'function') {
      await mod.run();
    } else {
      console.log(`${file}: No run() exported`);
    }
  } catch (err) {
    console.error(`${file}: ERROR`);
    console.error(err);
  }
}
