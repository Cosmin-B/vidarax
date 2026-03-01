# Vidarax Pitch Deck — 4 Slides, 3 Minutes

Style: Terminal Green or Neon Cyber. Dark. Technical. No fluff.

---

## SLIDE 1: The Problem (0:00 - 0:30)

**Title:** Video is the last unstructured frontier

**Visual:** A single big stat or graphic

**Content:**
- 3.5 trillion hours of video produced per year
- Less than 1% is searchable or analyzed
- Sending 1 hour to a frontier model: $2-4. Running 24/7: $28K-$350K per camera per year.
- Current options: cloud-only APIs (Twelve Labs, $157M raised) or dumb pixel-diff (Ring doorbell)
- Nothing exists that's open-source, on-device, and intelligent

**SAY:**
"The world produces 3.5 trillion hours of video every year. Less than 1% is searchable. You can send video to Gemini for $2 an hour — but run that 24/7 across a hundred cameras and you're looking at $350K a year. The alternative is Ring's motion detection, which alerts you every time a tree branch moves. There's nothing in between. No open-source, on-device video intelligence engine. Until now."

---

## SLIDE 2: What Vidarax Does (0:30 - 1:00)

**Title:** Any stream in, semantic events out

**Visual:** The architecture diagram:
```
Sources -> [Gate Engine] -> [Tiered VLM] -> [SpacetimeDB] -> Clients
            42ns/frame      2B->8B          real-time        Vue UI
            zero alloc      fallback        queryable        SDK
```

**Content:**
- Pass 1: Deterministic gate engine — scene cuts, artifacts, novelty at O(1) per frame. No ML. 42 nanoseconds.
- Pass 2: Tiered VLM routing — fast model (2B, 200ms) for easy frames, big model (8B) only when uncertain
- Pass 3: Structured events written to SpacetimeDB in real-time. Instantly queryable across all cameras.
- Result: 95% cost reduction vs sending everything to a cloud model

**SAY:**
"Vidarax is a three-pass architecture. Pass one: a deterministic gate engine — pure math, no ML — detects scene cuts, artifacts, and novelty at 42 nanoseconds per frame. It eliminates 95% of frames. Pass two: only the interesting frames hit a VLM. We run Qwen 2B at 200ms per chunk. If confidence is low, we escalate to 8B. Intelligent tiered routing. Pass three: structured events stream into SpacetimeDB in real-time, instantly queryable across all your video sources. The result: video intelligence at 5% of the cost of sending everything to Gemini."

---

## SLIDE 3: Live Demo (1:00 - 2:30)

**Title:** [LIVE DEMO]

**No slide content — this is the live demo portion**

**Demo flow:**
1. Show Dashboard — active runs, metrics
2. Upload 10-second video -> processed in 1.5 seconds
3. Show Run Detail — markers, keyframes, VLM descriptions
4. Show Tracing — pipeline flow, per-stage latency
5. Show Settings — tiered model routing, confidence threshold

**SAY (while demoing):**
"Let me show you. [Upload video] Ten-second video going in. [Wait 1.5s] Done. 1.5 seconds. 242 frames decoded, 20 markers extracted. [Click marker] Every marker has a keyframe and a semantic description. [Show tracing] Every frame is traceable from ingest through the gate engine to the VLM. 42 nanosecond p95 for the gate engine — that's lock-free, zero-allocation Rust. [Show settings] And here's the model routing — 2B for speed, 8B for accuracy, with an adjustable confidence threshold."

---

## SLIDE 4: Why This Matters (2:30 - 3:00)

**Title:** The Kafka of Video

**Visual:** Competitive positioning + key numbers

**Content:**
- Open-source. Self-hosted. Your data never leaves your infrastructure.
- 21,500 lines of Rust. Gate engine at 42ns/frame. Tiered VLM routing. WebRTC streaming. SpacetimeDB persistence.
- Built-in auto-distillation: vidarax learns YOUR cameras. Trains a tiny specialist model (500M) overnight on your own hardware.
- Use cases: gameplay QA, surveillance, manufacturing inspection, robotics
- Edge: runs on a $250 Jetson. Gate engine runs on a $80 Raspberry Pi.

**Key numbers:**
- Gate engine p95: 42ns/frame
- 10s video processing: 1.5s (6.7x real-time)
- API workflow p95: 2.99ms
- Cost vs cloud: 95% reduction
- Edge hardware: $250 (Jetson)

**SAY:**
"Vidarax is the Kafka of video. Kafka didn't compete with databases — it's the intelligent routing layer between them. Vidarax doesn't compete with VLMs — it makes them 20x more efficient by deciding which frames deserve attention. It's open-source, self-hosted, and runs on a $250 Jetson. No data leaves your infrastructure. We built this in Rust because video intelligence at the edge needs nanosecond-level processing, and we think every camera in the world deserves to be smart without a cloud subscription."

---

## Q&A Prep (1 minute)

**"How is this different from Twelve Labs?"**
"They're cloud-only, closed-source, $157M raised. We're open-source, self-hosted, edge-first. Different markets — they serve media companies, we serve anyone who can't send video to the cloud."

**"Why not just use Gemini with 1M token context?"**
"You can — for one video. But 100 cameras 24/7 = $350K/year in API calls. Our gate engine eliminates 95% of those calls."

**"What's the moat?"**
"Framework ownership. If developers import vidarax to build video AI, switching means rewriting. Plus auto-distillation: every deployment gets smarter over time with zero user effort."

**"Business model?"**
"Open-source engine, managed cloud for those who don't want to self-host. Per-event pricing. Vercel for video AI."
