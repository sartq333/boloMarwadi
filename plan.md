# BoloMarwadi — Project Plan

## Vision

A conversational AI character that hears you speak in English or Hindi, generates a witty response in Marwadi, and speaks it aloud through an animated Rajasthani dog character whose mouth moves in sync with the audio.

---

## Core Technology Decisions (Updated)

### (a) Marwadi Text Generation
**Status:** Feasible with prompt engineering; no dedicated model exists.

- Marwadi is a low-resource dialect of Rajasthani, sharing Devanagari script and ~75% vocabulary overlap with Hindi
- Large multilingual models run via `mlx-lm` (Llama 3.2, Qwen 2.5) have incidental Marwadi training exposure
- **Practical approach:** Strong multilingual model + detailed Marwadi system prompt with vocabulary substitutions, grammar rules (pronouns like *म्हारो/थारो* instead of *मेरा/तुम्हारा*), and few-shot examples
- **Expected quality:** Hindi-Marwadi hybrid — authentic to how diaspora Marwadi speakers actually communicate

Candidate models (MLX-compatible via `mlx-lm`):

| Model | Size | Notes |
|---|---|---|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4GB | Best multilingual/Indic coverage; recommended |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~2GB | Lighter; good for iteration |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4GB | Decent Hindi base |

### (b) STT — Voxtral Mini 4B Realtime (replaces mlx-whisper)

`mlx-whisper` is a batch processor — it waits for a complete audio clip and can take 10–30s. The replacement:

**[Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)** via `mlx-audio`:
- Supports **Hindi natively** (one of 13 languages: EN, ZH, HI, ES, AR, FR, PT, RU, DE, JA, KO, IT, NL)
- Streaming transcription with **<500ms latency** (configurable 240ms–2.4s window)
- MLX-native via the `mlx-audio` library (pip installable)
- Apache 2.0 license

### (c) TTS — AI4Bharat Indic-TTS with Rajasthani model

**The breakthrough finding:** AI4Bharat's original [Indic-TTS](https://github.com/AI4Bharat/Indic-TTS) project explicitly supports **Rajasthani** (one of 13 languages). This is the only known TTS model for the Rajasthani macrolanguage, which includes Marwadi as its dominant dialect.

- Architecture: FastPitch (acoustic model) + HiFi-GAN V1 (vocoder)
- Checkpoints: [GitHub releases v1](https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release)
- Underlying framework: Coqui TTS (PyTorch) — **the one non-MLX component**
- Note: AI4Bharat's newer models (Parler-TTS 2024, IndicF5 2025) dropped Rajasthani; the original is the only option

**MLX port plan (Phase 5):** FastPitch is a transformer-based model and HiFi-GAN is a convolutional net — both are portable to MLX following the same weight-conversion pattern shown in [MLX LLaMA inference docs](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html):
1. Load PyTorch state dict from AI4Bharat checkpoint
2. Map layer names to MLX equivalents
3. Re-implement FastPitch + HiFi-GAN in `mlx.nn`
4. Load converted NPZ weights

This makes the entire pipeline MLX-native in Phase 5.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      User Input                            │
│                   (English or Hindi)                       │
└─────────────────────────┬──────────────────────────────────┘
                          │ microphone audio stream
                          ▼
                ┌──────────────────────┐
                │  Voxtral Mini 4B     │  (Streaming STT)
                │  Realtime            │  via mlx-audio
                │  <500ms latency      │  Hindi + English
                └──────────┬───────────┘
                           │ transcript text (streamed tokens)
                           ▼
                ┌──────────────────────┐
                │  mlx-lm              │  (Language Model)
                │  Qwen2.5-7B or       │  Marwadi system prompt
                │  Llama-3.2-3B        │  + conversation history
                └──────────┬───────────┘
                           │ Marwadi text response
                           ▼
                ┌──────────────────────┐
                │  AI4Bharat           │  (TTS)
                │  Indic-TTS           │  Rajasthani voice model
                │  Rajasthani model    │  FastPitch + HiFi-GAN
                └──────────┬───────────┘
                           │ audio (.wav)
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
  ┌──────────────────┐        ┌─────────────────────┐
  │ amplitude        │        │  play audio to       │
  │ envelope         │        │  speakers            │
  │ extractor        │        └─────────────────────┘
  └────────┬─────────┘
           │ mouth signal (0.0–1.0 at ~30fps)
           ▼
  ┌──────────────────────────────────────────────┐
  │              Character UI                    │
  │  Rajasthani dog + fort background            │
  │  mouth frames synced to audio amplitude      │
  └──────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Pipeline Backend (`src/pipeline/`)

| File | Responsibility |
|---|---|
| `stt.py` | Wrap Voxtral Realtime via `mlx-audio`; stream transcript tokens; detect language |
| `llm.py` | Wrap `mlx-lm`; maintain conversation history; inject Marwadi prompt |
| `tts.py` | Wrap AI4Bharat Indic-TTS Rajasthani model; return `.wav` path |
| `audio_analysis.py` | Extract amplitude envelope from wav → `List[float]` at ~30fps for mouth sync |
| `pipeline.py` | Orchestrate full flow; expose via local WebSocket server |

### 2. Marwadi System Prompt (`src/prompts/marwadi_system.md`)

Critical for output quality. Must include:
- **Vocabulary substitutions:** Common Hindi → Marwadi (e.g., *क्या → के*, *नहीं → क틀ना*, *पानी → पाणी*, *आप → थे*, *मैं → म्हैं*)
- **Grammar rules:** Second-person pronouns (*थे/थारो/थारी*), verb endings (*-सो/-सी* for present tense, *-जो* for imperatives)
- **Tone guide:** Warm, witty, slightly formal — like a friendly elder from Jodhpur
- **Few-shot examples:** 10+ Q&A pairs demonstrating the target register
- **Hard constraint:** Always reply in Marwadi regardless of input language

### 3. Character UI

#### Option A: Web UI (Recommended to start)
**Tech:** Python FastAPI + WebSocket + HTML/CSS/JS canvas

- FastAPI backend serves the page and streams amplitude data via WebSocket
- Frontend renders character sprite on `<canvas>`, switches mouth frame based on amplitude
- Background: Rajasthani fort illustration (from provided assets)
- Fast to build; no external engine; easy to iterate

#### Option B: Godot 4 (For final polish)
**Tech:** Godot 4 scene + Python backend as subprocess

- Godot scene: 2D sprite with 3 mouth states (`mouth_closed`, `mouth_mid`, `mouth_open`)
- GDScript polls Python backend WebSocket for amplitude signal
- Better for: idle animations, particle effects, turban sway, eye blinks

**Recommended path:** Web UI for Phases 1–4, optional Godot migration in Phase 5+.

### 4. Character Sprite Preparation

The provided dog character needs these sprite variants created (Photoshop/Figma/Procreate):
- **Idle** — mouth closed, relaxed expression
- **Mouth mid** — slight opening
- **Mouth open** — speaking, wider opening
- **Thinking** (optional) — eyes looking up, while LLM generates

---

## Development Phases

### Phase 1: Core Pipeline — text in, text out (Days 1–2)
- [ ] Set up Python env: `mlx`, `mlx-lm`, `mlx-audio`
- [ ] Draft Marwadi system prompt; test with `mlx-lm` CLI; iterate on output quality
- [ ] Evaluate Qwen2.5-7B vs Llama-3.2-3B for Marwadi quality/speed tradeoff

### Phase 2: Voice In (Days 2–3)
- [ ] Set up Voxtral Mini 4B Realtime via `mlx-audio`
- [ ] Record mic → streaming Hindi/English transcript
- [ ] Wire transcript into LLM → Marwadi text output
- [ ] CLI demo: speak in Hindi → get Marwadi text response

### Phase 3: Voice Out — AI4Bharat Rajasthani TTS (Days 3–5)
- [ ] Download AI4Bharat Indic-TTS Rajasthani checkpoint from GitHub releases
- [ ] Install Coqui TTS; run inference with Rajasthani model
- [ ] Evaluate voice quality; verify Marwadi Devanagari text sounds correct
- [ ] `audio_analysis.py`: extract amplitude envelope for mouth sync
- [ ] Full CLI demo: speak Hindi → hear Marwadi reply

### Phase 4: Character Web UI (Days 5–9)
- [ ] Prepare 3-frame character sprite sheet (idle / mid / open mouth)
- [ ] FastAPI + WebSocket server
- [ ] Canvas sprite renderer in JS; fort background layer
- [ ] Wire WebSocket amplitude stream → mouth frame selection
- [ ] End-to-end test: full voice conversation with animated character

### Phase 5: MLX TTS Port — make it fully MLX-native (Days 9–14, stretch)
- [ ] Study AI4Bharat FastPitch architecture (examine their Coqui model config)
- [ ] Re-implement FastPitch in `mlx.nn` (transformer encoder + duration predictor + decoder)
- [ ] Re-implement HiFi-GAN vocoder in `mlx.nn` (convolutional generator)
- [ ] Convert PyTorch checkpoint weights → NPZ via name-mapping (follow MLX LLaMA pattern)
- [ ] Verify audio output matches original Coqui inference
- [ ] Replace `tts.py` Coqui call with MLX inference

### Phase 6: Polish & Godot (Optional, Days 14+)
- [ ] Multi-turn conversation history in LLM module
- [ ] Idle animations: eye blink, turban sway, body bob
- [ ] Godot 4 scene with 2D rig + mouth bone
- [ ] Interrupt / stop-speaking button
- [ ] Loading animation while LLM generates

---

## Project Structure

```
boloMarwadi/
├── plan.md
├── src/
│   ├── pipeline/
│   │   ├── stt.py               # Voxtral Realtime via mlx-audio
│   │   ├── llm.py               # mlx-lm + Marwadi prompt
│   │   ├── tts.py               # AI4Bharat Indic-TTS (Coqui → MLX in Phase 5)
│   │   ├── audio_analysis.py    # amplitude envelope extractor
│   │   └── pipeline.py          # orchestrator + WebSocket server
│   ├── prompts/
│   │   └── marwadi_system.md    # system prompt with vocab + grammar
│   ├── server/
│   │   └── app.py               # FastAPI + WebSocket
│   └── ui/
│       ├── index.html
│       ├── style.css
│       └── character.js         # canvas sprite + mouth sync
├── assets/
│   ├── character/               # sprite frames (idle, mid, open, thinking)
│   └── background/              # fort illustration
├── mlx_tts/                     # Phase 5: MLX port of FastPitch + HiFi-GAN
│   ├── fastpitch.py
│   ├── hifigan.py
│   └── convert_weights.py
├── godot/                       # Phase 6 (optional)
│   └── BoloMarwadi.godot
└── requirements.txt
```

---

## Dependencies

```
# requirements.txt

# MLX core
mlx
mlx-lm
mlx-audio          # Voxtral STT + Kokoro TTS (all MLX-native)

# AI4Bharat Indic-TTS (TTS, Phase 3 — only non-MLX component)
TTS                # Coqui TTS

# Server + UI
fastapi
uvicorn[standard]
websockets

# Audio I/O
sounddevice
numpy
scipy
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM generates Hindi instead of Marwadi | Strengthen few-shot examples; add post-processing vocabulary substitution layer |
| Voxtral misrecognizes Hinglish (Hindi+English mix) | Fall back to `mlx-whisper` as batch processor for problematic inputs |
| AI4Bharat Rajasthani TTS sounds too generic | Acceptable for v1; Phase 5 MLX port allows fine-tuning on Marwadi samples |
| FastPitch → MLX port is harder than expected | Keep Coqui as production path; MLX port becomes optional optimization |
| Mouth sync feels laggy | Pre-compute full amplitude envelope before starting playback; ring-buffer 100ms ahead |
| Character art quality | Start with simple 3-frame sprite; iterate art separately from pipeline |

---

## Open Questions

1. **Character rights:** Do you own the dog character art, or does it need to be redrawn for a distributable build?
2. **Interaction mode:** Push-to-talk button, or continuous VAD (voice activity detection)?
3. **Persona:** Should the character have a fixed name (e.g., "Moti Bhai", "Bhura Seth") with a personality?
4. **Platform:** Local Mac only, or distributable to others (affects whether non-MLX Coqui TTS is acceptable)?
