import assert from 'assert';
import { dataUtils } from '../src/data-utils.js';

export async function run() {
  const { generateRandomData, parseCSV, formatGeneratedDataToCSV } = dataUtils;

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
