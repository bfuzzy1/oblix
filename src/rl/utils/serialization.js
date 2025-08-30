export function mapToObject(map) {
  return Object.fromEntries(
    Array.from(map.entries()).map(([k, v]) => [k, Array.from(v)])
  );
}

export function objectToMap(obj = {}, ArrayType = Float32Array) {
  const map = new Map();
  for (const [k, v] of Object.entries(obj)) {
    map.set(k, new ArrayType(v));
  }
  return map;
}
