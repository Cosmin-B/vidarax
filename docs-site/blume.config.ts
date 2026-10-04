import { defineConfig } from "blume";

const page = (label: string, href: string) => ({ label, href });

export default defineConfig({
  title: "vidarax",
  description: "Self-hosted video intelligence engine. Streams in, structured events out.",
  deployment: {
    site: "https://vidarax.cosminbararu.com",
    base: "/docs",
  },
  content: {
    root: "src/content/docs",
  },
  feedback: false,
  github: {
    owner: "Cosmin-B",
    repo: "vidarax",
    branch: "main",
    dir: "docs-site",
  },
  navigation: {
    sidebar: {
      display: "group",
      items: [
        page("What is vidarax", "/"),
        page("Quickstart", "/quickstart"),
        page("Agent workflows", "/agents"),
        page("Architecture", "/architecture"),
        page("Ingest", "/ingest"),
        page("Local audio", "/audio"),
        page("Gemini Flash review", "/gemini-flash"),
        page("Mage-VL debug", "/mage-vl"),
        page("Per-frame filter", "/gate"),
        page("API reference", "/api"),
        page("Events and SDK", "/events"),
        page("Policy rollouts", "/policy-rollouts"),
        page("Trigger programs", "/triggers"),
        page("Edge deployment", "/edge"),
        page("Operations", "/operations"),
        page("Development", "/development"),
        {
          label: "Internals",
          items: [
            page("Media plane", "/internals/media-plane"),
            page("Decode sidecar", "/internals/decode-sidecar"),
            page("Filter internals", "/internals/gate-internals"),
            page("State and cancellation", "/internals/state-and-cancellation"),
            page("WAL and events", "/internals/wal-and-events"),
            page("WebRTC ingest", "/internals/webrtc-ingest"),
            page("Allocation discipline", "/internals/allocation-discipline"),
          ],
        },
      ],
    },
  },
  theme: {
    mode: "dark",
    accent: { dark: "#4fc07d", light: "#2f7c50" },
    background: { dark: "#0d1014", light: "#faf7f1" },
    radius: "sm",
  },
});
