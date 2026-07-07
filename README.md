# Multimodal Video RAG Assistant

> Upload lecture / demo / lab videos, ask questions in natural language, and get answers **with timestamp citations** — click a citation to jump to that exact moment in the video. **Fully local, no API keys.**

<!-- Record a short screen capture (ask a question → click a timestamp → the player jumps) and drop it here: -->
![demo](docs/demo.gif)

---

## Why this is more than a "chat with your PDF"

Text-only RAG indexes text. **Video carries three signals: a spoken transcript, visual frames, and time alignment.** This project builds a **multimodal index** that binds transcript chunks *and* keyframe embeddings to a shared **timestamp timeline**, and retrieves across both modalities using CLIP's shared image–text space — so the system doesn't just answer, it **locates the answer in time** and can be scoped to a single video or a whole course.

**Example questions**
- "When is `temporal consistency` discussed?"
- "什么时候讲到 permute?" *(multilingual — Chinese question over English lectures works)*
- "Which lecture in this course covers hashing?" *(course-wide search across many videos)*
- "Summarize minutes 3–5 and give timestamps."

## Features

- 🎙️ **Transcription** with word-accurate timestamps (faster-whisper, GPU).
- 🖼️ **Keyframe extraction** (content-aware scene detection with a uniform-sampling fallback for low-cut lectures).
- 🔎 **Cross-modal retrieval** — one question → BGE text search over the transcript **and** CLIP text→image search over keyframes.
- 🥇 **Cross-encoder reranking** (bge-reranker-v2-m3) — measured to lift ranking quality (see [Evaluation](#evaluation)).
- 🗂️ **Course / video scoping** — one shared vector store, isolated by `course_id` / `video_id` metadata (multi-tenancy, not a DB per course).
- 💬 **Grounded answers with timestamp citations** via a local LLM (Ollama).
- 🖥️ **Web UI** — a live, auto-scrolling transcript synced to playback; click any line or citation to seek; batch-ingest whole course folders.
- 📊 **A real evaluation harness** — Recall@k / MRR, dense vs. reranked.

## Architecture

**Ingestion**
```
video ─ffmpeg─┬─ audio ─faster-whisper─► transcript segments (timestamps)
              │                            └─ chunk ─► BGE-m3 text embed ─┐
              └─ frames ─scene/uniform──► keyframes (timestamps)          ├─► LanceDB
                                           └─ OpenCLIP image embed ───────┘   (2 tables + course_id / video_id / timestamp)
```

**Query**
```
question ─┬─ BGE-m3 ────► transcript index  (dense top-20) ─► bge-reranker (cross-encoder) ─► top-6 ─┐
          └─ CLIP text ─► keyframe index ─────────────────────────────────────────────────► top-k ──┤
                                                                     grounded prompt ◄───────────────┘
                                                                          │
                                                          Ollama ─► answer + timestamp citations
```

## Evaluation

Retrieval is measured with a **synthetic eval set**: the LLM generates a question from each transcript chunk, and that chunk is the ground truth. Retrieval is scoped per course (how the app runs).

On **40 queries over a 6,768-chunk / 30-video / 2-course corpus:**

| method          | Recall@6 | MRR   |
| --------------- | :------: | :---: |
| dense           |   1.00   | 0.86  |
| **dense + rerank** | **1.00** | **0.955** |

Reranking pushes the correct chunk to rank **#1 for ~91% of queries** (vs. ~72% for dense) — MRR **0.86 → 0.955**. Recall is saturated here because synthetic questions are lexically close to their source chunk; harder paraphrased queries would also surface reranking's *recall* benefit. (Honest caveat baked into the write-up — see [`docs/design-and-interview.md`](docs/design-and-interview.md).)

Reproduce:
```bash
python scripts/build_eval.py 40   # generate the eval set (uses the local LLM)
python scripts/evaluate.py        # Recall@k / MRR, dense vs. rerank
```

## Tech stack

**Fully local / open-source — clone, install, run, no API keys.**

| Stage | Choice |
| --- | --- |
| Transcription | **faster-whisper** (CTranslate2, GPU) |
| Keyframes | PySceneDetect + OpenCV |
| Text embeddings | **BGE-m3** (multilingual) via sentence-transformers |
| Visual / cross-modal | **OpenCLIP** ViT-B/32 |
| Reranking | **bge-reranker-v2-m3** cross-encoder |
| Vector store | **LanceDB** (embedded, 2 tables, metadata scoping) |
| Generation | **Ollama** (local LLM, e.g. Qwen) |
| UI / config | Streamlit · pydantic-settings |

## Quickstart

Requires an NVIDIA GPU, Python ≥ 3.10, `ffmpeg`, and [Ollama](https://ollama.com).

```bash
# 1. environment (Python 3.11)
conda create -n mvrag python=3.11 -y && conda activate mvrag

# 2. install — GPU torch first (faster-whisper's CTranslate2 needs CUDA 12), then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[ml]"

# 3. local LLM (pick any Ollama model; override the default in a .env file)
ollama pull qwen2.5:7b
#   echo 'MVRAG_OLLAMA_MODEL=qwen2.5:7b' > .env

# 4. sanity check
python scripts/check_env.py        # expect: Python ✅  ffmpeg ✅  torch+CUDA ✅  Ollama ✅

# 5. ingest a video (2nd arg = a course tag you choose)
python scripts/ingest.py path/to/lecture01.mp4 MyCourse
python scripts/index.py  path/to/lecture01.mp4 MyCourse

# 6a. ask from the CLI
python scripts/ask.py "how do you reshape a tensor" --course MyCourse

# 6b. …or launch the web UI
streamlit run app/streamlit_app.py
```

### Adding courses in bulk

Organize videos as **one folder per course inside `data/`** — the **folder name becomes the `course_id`**:

```
data/
├── CS61A/                 ← folder name = course tag
│   ├── lecture01.mp4
│   ├── lecture02.mp4
│   └── …
└── CS106B/
    ├── week1_intro.mp4
    └── …
```

Then in the web UI sidebar, **"Batch ingest folders"** will show `Found N videos in: CS61A, CS106B` → click **"Ingest all folders"**. It walks every `data/<Course>/` folder, ingests each video (tagged with the folder name), and **skips any already done** — so it is safe to re-run or resume after an interruption.

- Videos must sit **directly** in the course folder (not nested deeper).
- Supported extensions: `.mp4`, `.mkv`, `.mov`, `.webm`.
- Artifacts (`data/<course>__<video>/`) and the vector store are created automatically and are git-ignored.

Equivalent from the CLI:
```bash
for course in CS61A CS106B; do
  for f in "data/$course"/*.mp4; do
    python scripts/ingest.py "$f" "$course" && python scripts/index.py "$f" "$course"
  done
done
```

## How a query flows

1. The question is embedded twice — by **BGE-m3** (to search the transcript index) and by **CLIP's text encoder** (to search the keyframe index), because the two modalities live in different embedding spaces.
2. The transcript index returns 20 dense candidates; a **cross-encoder reranks** them to the top 6.
3. The top chunks — each labelled with its `[mm:ss]` span — are placed in a **grounded prompt**; the LLM answers **only** from them and cites the timestamp(s) it used.
4. The UI turns each citation into a button that seeks the player to that moment.

## Project structure

```
src/mvrag/
  config.py          # centralized, typed config (pydantic-settings)
  schemas.py         # data models (all time-anchored, course/video-scoped)
  ingestion/         # audio→Whisper transcript · frames→scene/uniform keyframes
  embedding/         # BGE text · OpenCLIP image+text
  indexing/          # transcript chunking · LanceDB store (scoped search)
  retrieval/         # cross-modal retriever · cross-encoder reranker
  generation/        # grounded answering with timestamp citations
app/streamlit_app.py # web UI (live-synced transcript, jumps, batch ingest)
scripts/             # ingest · index · ask · rerank_demo · build_eval · evaluate
docs/                # design notes + interview Q&A
```

## Design decisions & deep dive

[`docs/design-and-interview.md`](docs/design-and-interview.md) documents every key decision (faster-whisper vs. Whisper, LanceDB vs. FAISS/Chroma, bi- vs. cross-encoder, multi-tenancy via metadata filtering, why **no fine-tuning**, chunk-size trade-offs, …) and the empirical findings from building it — including a few bugs found and fixed by *measuring* rather than guessing.

## Known limitations & future work

- **English-oriented visual retrieval** — OpenCLIP is weak on non-English text-in-image and code screencasts → swap in Chinese-CLIP / multilingual-CLIP.
- **Synthetic eval is optimistic** — add hand-written paraphrased/indirect queries to stress-test recall.
- **Segment-level timestamps** — word-level alignment (whisperX) would sharpen citations.
- **Retrieval** — add hybrid (dense + BM25) to catch literal terms dense embeddings miss; add an LLM-as-judge for answer-quality (groundedness / citation) scoring.
- **Ops** — a FastAPI backend + async ingestion queue, and a Docker image (GPU + Ollama) for one-command reproducibility.

---

*A portfolio project exploring multimodal RAG over video — transcript + visual + time, retrieved together and cited back to the moment.*
