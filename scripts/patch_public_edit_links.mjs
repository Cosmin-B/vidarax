import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputRoot = path.join(repositoryRoot, 'docs-site', 'dist');
const replacements = [
  {
    from: 'https://github.com/Cosmin-B/vidarax/edit/main/docs-site/src/content/docs/quickstart.mdx',
    to: 'https://github.com/Cosmin-B/vidarax/edit/main/docs-site/src/content/docs/quickstart.mdoc',
  },
  {
    from: 'https://github.com/Cosmin-B/vidarax/edit/main/docs-site/src/content/docs/operations.mdx',
    to: 'https://github.com/Cosmin-B/vidarax/edit/main/docs-site/src/content/docs/operations.mdoc',
  },
];
const textExtensions = new Set(['.html', '.json', '.md', '.mdx', '.txt']);
const files = await collectFiles(outputRoot);
const found = new Set();
let updatedFiles = 0;

for (const file of files) {
  if (!textExtensions.has(path.extname(file))) continue;
  const before = await readFile(file, 'utf8');
  let after = before;
  for (const replacement of replacements) {
    if (!after.includes(replacement.from)) continue;
    found.add(replacement.from);
    after = after.split(replacement.from).join(replacement.to);
  }
  if (after !== before) {
    await writeFile(file, after, 'utf8');
    updatedFiles += 1;
  }
}

for (const replacement of replacements) {
  if (!found.has(replacement.from)) {
    throw new Error(`Expected Blume edit link was not emitted: ${replacement.from}`);
  }
}

console.log(`Repointed Vidarax edit actions to the two published MDOC files in ${updatedFiles} output file(s).`);

async function collectFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await collectFiles(target)));
    else if (entry.isFile()) files.push(target);
  }
  return files;
}
