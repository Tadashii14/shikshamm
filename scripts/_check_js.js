// Syntax-check inline <script> blocks in a template using node's vm parser.
const fs = require('fs');
const vm = require('vm');
const files = process.argv.slice(2);
let failed = false;
for (const f of files) {
  const html = fs.readFileSync(f, 'utf8');
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  let m, i = 0;
  while ((m = re.exec(html)) !== null) {
    i++;
    try {
      new vm.Script(m[1], { filename: `${f}#script${i}` });
      console.log(`OK   ${f} #script${i} (${m[1].length} chars)`);
    } catch (e) {
      failed = true;
      console.log(`FAIL ${f} #script${i}: ${e.message}`);
      const lines = m[1].split('\n');
      const ln = (e.stack.match(/<anonymous>:(\d+)/) || [])[1];
      if (ln) {
        const n = parseInt(ln, 10);
        for (let k = Math.max(0, n - 3); k < Math.min(lines.length, n + 2); k++) {
          console.log(`   ${k + 1}: ${lines[k]}`);
        }
      }
    }
  }
}
process.exit(failed ? 1 : 0);
