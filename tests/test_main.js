import assert from 'assert';
import fs from 'fs';
import path from 'path';

function extractFunction(code, name) {
  const startToken = `function ${name}`;
  const startIdx = code.indexOf(startToken);
  if (startIdx === -1) throw new Error(`Function ${name} not found`);
  let i = code.indexOf('{', startIdx);
  let count = 1;
  while (count > 0 && ++i < code.length) {
    const ch = code[i];
    if (ch === '{') count++;
    else if (ch === '}') count--;
  }
  const fnStr = code.slice(startIdx, i + 1);
  return eval(`(${fnStr})`);
}

/**
 *
 */
export async function run() {
  const code = fs.readFileSync(path.join('src', 'main.js'), 'utf8');
  const generateRandomData = extractFunction(code, 'generateRandomData');
  const parseCSV = extractFunction(code, 'parseCSV');
  const formatGeneratedDataToCSV = extractFunction(code, 'formatGeneratedDataToCSV');

  const csvStr = generateRandomData(5, 2, 1, 0);
  const rows = csvStr.split('\n');
  assert.strictEqual(rows.length, 5);
  for (const row of rows) {
    const vals = row.split(',').map(v => parseFloat(v));
    assert.strictEqual(vals.length, 3);
    for (const v of vals) {
      assert.ok(v >= 0.01 && v <= 0.99);
    }
  }

  const sampleCsv = '0.1,0.2,1\n0.3,0.4,0';
  const parsed = parseCSV(sampleCsv);
  assert.deepStrictEqual(parsed, [
    { input: [0.1, 0.2], output: [1] },
    { input: [0.3, 0.4], output: [0] }
  ]);

  const formatted = formatGeneratedDataToCSV(parsed);
  assert.deepStrictEqual(formatted, [
    '0.100, 0.200, 1.000',
    '0.300, 0.400, 0.000'
  ]);
}
