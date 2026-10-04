import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsRoot = path.join(repositoryRoot, 'docs-site', 'src', 'content', 'docs');
const docsSiteRoot = path.join(repositoryRoot, 'docs-site');
const manifest = JSON.parse(await readFile(path.join(docsSiteRoot, 'source-manifest.json'), 'utf8'));

test('the pinned Vidarax pages keep Blume routes and converted executable examples', async () => {
  assert.equal(manifest.files.length, 22);
  const currentPages = await collectMarkdown(docsRoot);
  const currentPagePaths = new Set(currentPages);

  for (const entry of manifest.files) {
    const relative = entry.path.slice('docs-site/src/content/docs/'.length).replace(/\.mdoc$/, '.mdx');
    const currentPath = path.join(docsRoot, relative);
    assert.ok(currentPagePaths.has(currentPath), `missing pinned page: ${relative}`);
    const current = await readFile(currentPath, 'utf8');
    assert.doesNotMatch(withoutFences(current), /\]\(\/docs(?:\/|[#)])/);
    if (entry.route) assert.match(current, new RegExp(`^---\\n[\\s\\S]*?\\nslug: ${escapeRegExp(entry.route)}\\n`));
    else assert.match(current, /^---\n[\s\S]*?\n---\n/);
    const original = execFileSync('git', ['show', `${manifest.revision}:${entry.path}`], {
      cwd: repositoryRoot,
      encoding: 'utf8',
    });
    assert.equal(createHash('sha256').update(original).digest('hex'), entry.sha256, entry.path);
    if (entry.path.endsWith('.mdoc')) {
      assert.deepEqual(fencedPayloads(current), fencedPayloads(original), entry.path);
    }
  }

  const quickstart = await readFile(path.join(docsRoot, 'quickstart.mdx'), 'utf8');
  assert.match(quickstart, /:::note\[API keys are on by default\]/);
  assert.match(quickstart, /VIDARAX_REQUIRE_API_KEY=false/);
  const operations = await readFile(path.join(docsRoot, 'operations.mdx'), 'utf8');
  assert.match(operations, /:::warning\[Local compose boundaries\]/);
  assert.match(operations, /persistent WAL storage/);
  assert.equal(/\{%\s*aside/.test(quickstart + operations), false);

  const config = await readFile(path.join(docsSiteRoot, 'blume.config.ts'), 'utf8');
  assert.match(config, /base:\s*["']\/docs["']/);
  assert.equal(/basePath\s*:/.test(config), false);
  assert.match(config, /"\/quickstart"/);
  const packageJson = JSON.parse(await readFile(path.join(docsSiteRoot, 'package.json'), 'utf8'));
  assert.equal(packageJson.dependencies.blume, '2.0.3');
  assert.match(packageJson.scripts.build, /blume build/);
  assert.match(packageJson.scripts.dev, /blume dev/);
  assert.equal(packageJson.dependencies.astro, undefined);
  assert.equal(packageJson.dependencies['@astrojs/starlight'], undefined);
});

test('renamed MDX pages keep edit actions pointed at the public MDOC originals', async () => {
  execFileSync('npm', ['run', 'build'], { cwd: docsSiteRoot, encoding: 'utf8' });
  const packageJson = JSON.parse(await readFile(path.join(docsSiteRoot, 'package.json'), 'utf8'));
  assert.match(packageJson.scripts.build, /patch_public_edit_links\.mjs/);

  for (const filename of ['quickstart', 'operations']) {
    const publishedSource = `docs-site/src/content/docs/${filename}.mdoc`;
    execFileSync('git', ['cat-file', '-e', `${manifest.revision}:${publishedSource}`], {
      cwd: repositoryRoot,
      stdio: 'pipe',
    });

    const html = await readFile(path.join(docsSiteRoot, 'dist', filename, 'index.html'), 'utf8');
    assert.match(html, new RegExp(`href="https://github\\.com/Cosmin-B/vidarax/edit/main/${publishedSource.replaceAll('/', '\\/')}"`));
    assert.doesNotMatch(html, new RegExp(`href="https://github\\.com/Cosmin-B/vidarax/edit/main/${publishedSource.replace(/\.mdoc$/, '\\.mdx').replaceAll('/', '\\/')}"`));
  }
});

async function collectMarkdown(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) files.push(...(await collectMarkdown(target)));
    if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.mdx'))) files.push(target);
  }
  return files;
}

function fencedPayloads(markdown) {
  const payloads = [];
  let fence;
  let payload = [];
  for (const line of markdown.split(/\r?\n/)) {
    const opener = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (!fence && opener) {
      fence = opener[1];
      payload = [];
    } else if (fence && new RegExp(`^\\s{0,3}${fence[0]}{${fence.length},}\\s*$`).test(line)) {
      payloads.push(payload.join('\n'));
      fence = undefined;
    } else if (fence) {
      payload.push(line);
    }
  }
  return payloads;
}

function withoutFences(markdown) {
  const output = [];
  let fence;
  for (const line of markdown.split(/\r?\n/)) {
    const opener = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (!fence && opener) {
      fence = opener[1];
      continue;
    }
    if (fence) {
      if (new RegExp(`^\\s{0,3}${fence[0]}{${fence.length},}\\s*$`).test(line)) fence = undefined;
      continue;
    }
    output.push(line);
  }
  return output.join('\n');
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
