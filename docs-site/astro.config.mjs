// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import markdoc from '@astrojs/markdoc';

export default defineConfig({
  base: '/docs',
  integrations: [
    markdoc(),
    starlight({
      title: 'vidarax',
      description:
        'Self-hosted video intelligence engine. Streams in, structured events out.',
      customCss: ['./src/styles/theme.css'],
      // Default to the dark theme on first visit. Starlight follows the
      // system preference when no theme is stored; this stores dark before
      // the user has picked anything, and never overrides a stored choice.
      head: [
        {
          tag: 'script',
          content:
            "if (!localStorage.getItem('starlight-theme')) { localStorage.setItem('starlight-theme', 'dark'); document.documentElement.dataset.theme = 'dark'; }",
        },
      ],
      sidebar: [
        { label: 'What is vidarax', link: '/' },
        { label: 'Quickstart', slug: 'quickstart' },
        { label: 'Architecture', slug: 'architecture' },
        { label: 'Ingest', slug: 'ingest' },
        { label: 'The gate', slug: 'gate' },
        { label: 'API reference', slug: 'api' },
        { label: 'Events and SDK', slug: 'events' },
        { label: 'Operations', slug: 'operations' },
        { label: 'Development', slug: 'development' },
        {
          label: 'Internals',
          items: [
            { label: 'Media plane', slug: 'internals/media-plane' },
            { label: 'Decode sidecar', slug: 'internals/decode-sidecar' },
            { label: 'Gate internals', slug: 'internals/gate-internals' },
            {
              label: 'State and cancellation',
              slug: 'internals/state-and-cancellation',
            },
            { label: 'WAL and events', slug: 'internals/wal-and-events' },
            { label: 'WebRTC ingest', slug: 'internals/webrtc-ingest' },
            {
              label: 'Allocation discipline',
              slug: 'internals/allocation-discipline',
            },
          ],
        },
      ],
    }),
  ],
});
