import fs from 'fs';
import path from 'path';

const __dirname = path.dirname(new URL(import.meta.url).pathname);

console.log('🧪 Starting Oblix Performance Benchmarks\n');

// Run performance optimization benchmarks first
try {
  console.log('🚀 Running Performance Optimization Benchmarks...');
  const { runPerformanceBenchmarks } = await import('./bench_performance_optimizations.js');
  await runPerformanceBenchmarks();
  console.log('\n' + '='.repeat(60) + '\n');
} catch (err) {
  console.error('❌ Performance optimization benchmarks failed:', err.message);
}

// Run all other benchmarks
const files = fs
  .readdirSync(__dirname)
  .filter(f => f.endsWith('.js') && f !== 'run.js' && f !== 'bench_performance_optimizations.js')
  .sort();

console.log('📊 Running Standard Benchmarks...\n');

for (const file of files) {
  try {
    console.log(`🔍 Running ${file}...`);
    const mod = await import(`./${file}`);
    if (typeof mod.run === 'function') {
      await mod.run();
    } else {
      console.log(`  ⚠️  ${file}: No run() exported`);
    }
    console.log('');
  } catch (err) {
    console.error(`  ❌ ${file}: ERROR`);
    console.error(`     ${err.message}`);
  }
}

console.log('✅ All benchmarks completed!');
