export const dataUtils = {
  formatGeneratedDataToCSV: function (dataArray) {
    if (!Array.isArray(dataArray)) return [];
    return dataArray
      .map((sample) => {
        if (
          !sample ||
          !Array.isArray(sample.input) ||
          !Array.isArray(sample.output)
        ) {
          console.warn('Skipping invalid sample in formatGeneratedDataToCSV');
          return null;
        }
        return [...sample.input, ...sample.output]
          .map((v) => (typeof v === 'number' && isFinite(v) ? v.toFixed(3) : 'NaN'))
          .join(', ');
      })
      .filter((row) => row !== null);
  },

  generateRandomData: function (
    numSamples,
    numInputs,
    numOutputs = 1,
    noiseLevel = 0.05,
  ) {
    if (numInputs <= 0 || numOutputs <= 0) return '';
    const data = [];
    for (let i = 0; i < numSamples; i++) {
      const input = [];
      for (let j = 0; j < numInputs; j++) input.push(Math.random());
      const output = [];
      for (let j = 0; j < numOutputs; j++) {
        const base = Math.sin(input[0] * Math.PI * 2) * 0.4 + 0.5;
        const noise = (Math.random() - 0.5) * 2 * noiseLevel;
        let final = Math.max(0.01, Math.min(0.99, base + noise));
        output.push(final);
      }
      data.push([...input, ...output].map((v) => v.toFixed(3)).join(', '));
    }
    return data.join('\n');
  },

  parseCSV: function (csvString, numOutputs = 1) {
    if (!csvString || typeof csvString !== 'string') return [];
    const outputs = Math.max(1, Math.floor(numOutputs));
    return csvString
      .trim()
      .split('\n')
      .map((r) => r.trim())
      .filter((r) => r.length > 0)
      .map((r, idx) => {
        const vals = r.split(',').map((v) => parseFloat(v.trim()));
        if (vals.some(isNaN)) {
          console.warn(`R ${idx + 1} NaN`);
          return null;
        }
        if (vals.length < outputs + 1) {
          console.warn(`R ${idx + 1} insufficient vals`);
          return null;
        }
        const input = vals.slice(0, vals.length - outputs);
        const output = vals.slice(vals.length - outputs);
        if (input.length === 0 || output.length !== outputs) {
          console.warn(`R ${idx + 1} invalid io lens`);
          return null;
        }
        return { input: input, output: output };
      })
      .filter((i) => i !== null);
  },
};
