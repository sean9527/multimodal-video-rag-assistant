"""Streamlit UI for the Multimodal Video RAG Assistant.

Run from the project root:
    streamlit run app/streamlit_app.py

Pick a course (and optionally narrow to one video), ask questions, and click a
source / citation to jump the player to that exact moment — across videos when
asking a whole course. Requires Ollama running and at least one ingested video.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html

from mvrag.config import PROJECT_ROOT, settings
from mvrag.generation.answerer import answer
from mvrag.indexing.store import index_video
from mvrag.ingestion.pipeline import ingest_video, make_video_id
from mvrag.schemas import IngestionResult, format_timestamp

st.set_page_config(page_title="Video RAG Assistant", layout="wide")
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}


@st.cache_data(show_spinner=False)
def load_ingested() -> dict[str, dict]:
    """video_id -> serialized IngestionResult, for every ingested video."""
    out: dict[str, dict] = {}
    for ing in sorted(settings.data_dir.glob("*/ingestion.json")):
        r = IngestionResult.model_validate_json(ing.read_text(encoding="utf-8"))
        out[r.video_id] = r.model_dump()
    return out


def find_video_file(meta: dict) -> Path | None:
    for cand in (Path(meta["video_path"]), PROJECT_ROOT / meta["video_path"]):
        if cand.exists():
            return cand
    for h in settings.data_dir.glob(f"{meta['video_id']}.*"):
        if h.suffix.lower() in VIDEO_EXTS:
            return h
    return None


@st.cache_data(show_spinner=False)
def _video_data_uri(path_str: str) -> str:
    """Base64 data URI for the video, so it can live inside the transcript-sync
    HTML component (which needs the video in the SAME iframe to read playback
    time via JS). Cached per file. Best for small/medium videos."""
    return "data:video/mp4;base64," + base64.b64encode(Path(path_str).read_bytes()).decode()


def render_player(meta: dict, vfile: Path, start: float) -> None:
    """Self-contained HTML5 player + transcript that highlights and auto-scrolls
    to the current line as the video plays (client-side, via the `timeupdate`
    event). Click any line to seek. No caption overlay on the video itself."""
    cues = "".join(
        f'<div class="cue" data-s="{seg["start"]}" data-e="{seg["end"]}" '
        f'onclick="seekTo({seg["start"]})"><span class="ts">{format_timestamp(seg["start"])}</span> '
        f'{html.escape(seg["text"])}</div>'
        for seg in meta["segments"]
    )
    doc = f"""
    <style>
      body {{ margin:0; font-family: system-ui, sans-serif; }}
      video {{ width:100%; max-height:300px; background:#000; border-radius:8px; }}
      #tx {{ position:relative; height:230px; overflow-y:auto; margin-top:8px;
             border:1px solid #ccc; border-radius:8px; padding:4px; }}
      .cue {{ padding:3px 8px; cursor:pointer; border-radius:4px; line-height:1.35; }}
      .cue:hover {{ background:#eef; }}
      .cue.active {{ background:#2e7d32; color:#fff; }}
      .ts {{ color:#999; margin-right:6px; font-variant-numeric:tabular-nums; }}
      .cue.active .ts {{ color:#d9f2d9; }}
    </style>
    <video id="vid" controls src="{_video_data_uri(str(vfile))}"></video>
    <div id="tx">{cues}</div>
    <script>
      const vid = document.getElementById('vid'), tx = document.getElementById('tx');
      const cues = [...document.querySelectorAll('.cue')];
      function seekTo(t) {{ vid.currentTime = t; vid.play(); }}
      let last = null;
      function sync() {{
        const t = vid.currentTime; let act = null;
        for (const c of cues) {{
          const on = t >= parseFloat(c.dataset.s) && t < parseFloat(c.dataset.e);
          if (on) act = c;
          c.classList.toggle('active', on);
        }}
        if (act && act !== last) {{
          tx.scrollTop = act.offsetTop - tx.clientHeight / 2 + act.clientHeight / 2;
          last = act;
        }}
      }}
      vid.addEventListener('timeupdate', sync);
      vid.addEventListener('seeked', sync);
      const START = {int(start)};
      if (START > 0) vid.addEventListener('loadedmetadata', () => {{ vid.currentTime = START; }}, {{once: true}});
    </script>
    """
    st_html(doc, height=560)


def static_transcript(meta: dict, seek: float) -> None:
    """Scrollable transcript highlighting the segment at the current seek position.
    Used for large videos, where base64-embedding for the live-sync player is too
    heavy (a 1h lecture would be 100+ MB inlined into the iframe)."""
    rows = []
    for seg in meta["segments"]:
        ts = format_timestamp(seg["start"])
        text = html.escape(seg["text"])
        if seg["start"] <= seek <= seg["end"]:
            rows.append(f"<div style='background:#2e7d32;color:#fff;padding:3px 8px;"
                        f"border-radius:4px'>▶ <b>{ts}</b>&nbsp; {text}</div>")
        else:
            rows.append(f"<div style='padding:3px 8px'><span style='color:#888'>{ts}</span>"
                        f"&nbsp; {text}</div>")
    with st.container(height=240):
        st.markdown("".join(rows), unsafe_allow_html=True)


def to_seconds(mmss: str) -> int:
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def play(video_id: str, seconds: float = 0) -> None:
    st.session_state.play_video = video_id
    st.session_state.seek = int(seconds)
    st.rerun()


st.session_state.setdefault("seek", 0)
st.session_state.setdefault("play_video", None)
st.session_state.setdefault("last_resp", None)
st.session_state.setdefault("scope_key", None)

st.title("🎬 Multimodal Video RAG Assistant")

videos = load_ingested()
by_course: dict[str, list[str]] = {}
for vid, meta in videos.items():
    by_course.setdefault(meta["course_id"], []).append(vid)

# ---------------- sidebar: scope + upload ----------------
course_id = video_id = None
with st.sidebar:
    st.header("Scope")
    if by_course:
        course_id = st.selectbox("Course", sorted(by_course))
        pick = st.selectbox("Ask about", ["🎓 Whole course", *sorted(by_course[course_id])])
        video_id = None if pick == "🎓 Whole course" else pick
    else:
        st.info("No videos yet — upload one below.")

    st.divider()
    st.subheader("Add a video")
    up = st.file_uploader("Video file", type=[e[1:] for e in VIDEO_EXTS])
    up_course = st.text_input("Course id", value=course_id or "CSCE0000")
    if up is not None and st.button("Ingest + index", use_container_width=True):
        dest = settings.data_dir / up.name
        dest.write_bytes(up.getbuffer())
        with st.spinner("Transcribing → keyframes → embedding → indexing…"):
            res = ingest_video(dest, course_id=up_course)
            index_video(res)
        load_ingested.clear()
        st.success(f"Indexed '{res.video_id}' in course '{up_course}'")
        st.rerun()

    st.divider()
    st.subheader("Batch ingest folders")
    st.caption("Put videos in data/<CourseName>/ — ingests each (course = folder name), skipping done ones.")
    pairs = []
    for d in sorted(p for p in settings.data_dir.iterdir() if p.is_dir()):
        for v in sorted(d.iterdir()):
            if v.is_file() and v.suffix.lower() in VIDEO_EXTS:
                pairs.append((d.name, v))
    if pairs:
        st.write(f"Found **{len(pairs)}** videos in: {', '.join(sorted({c for c, _ in pairs}))}")
        if st.button("Ingest all folders", use_container_width=True):
            bar = st.progress(0.0, text="starting…")
            new = 0
            for i, (course, vpath) in enumerate(pairs, 1):
                bar.progress(i / len(pairs), text=f"[{i}/{len(pairs)}] {course} / {vpath.name}")
                if (settings.data_dir / make_video_id(course, vpath) / "ingestion.json").exists():
                    continue
                index_video(ingest_video(vpath, course_id=course))
                new += 1
            load_ingested.clear()
            st.success(f"Done — ingested {new} new video(s).")
            st.rerun()
    else:
        st.caption("(no data/<Course>/ folders with videos found)")

if not by_course:
    st.stop()

# reset transient state when the scope changes
scope_key = f"{course_id}/{video_id}"
if st.session_state.scope_key != scope_key:
    st.session_state.update(
        scope_key=scope_key, last_resp=None, seek=0,
        play_video=video_id or sorted(by_course[course_id])[0],
    )

# ---------------- main: player | chat ----------------
left, right = st.columns([3, 2], gap="large")

with left:
    meta = videos.get(st.session_state.play_video)
    vfile = find_video_file(meta) if meta else None
    if not vfile:
        st.warning("Video file not found on disk — keep it in data/ to enable playback.")
    elif vfile.stat().st_size <= 40 * 1024 * 1024:  # small clip -> base64 live-sync player
        render_player(meta, vfile, st.session_state.seek)
        st.caption(f"**{meta['video_id']}** · {format_timestamp(meta['duration'])} · course "
                   f"`{meta['course_id']}` · transcript auto-scrolls as it plays; click a line to seek")
    else:  # large video (e.g. full lecture) -> efficient native player + static transcript
        st.video(str(vfile), start_time=int(st.session_state.seek))
        st.caption(f"**{meta['video_id']}** · {format_timestamp(meta['duration'])} · course "
                   f"`{meta['course_id']}` · large video → transcript below highlights your position "
                   f"(live auto-scroll is only enabled for short clips)")
        static_transcript(meta, st.session_state.seek)

with right:
    st.caption(f"Asking: **{'the whole course' if video_id is None else video_id}**")
    query = st.text_input("Question", "how do you reshape a tensor")
    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("retrieving + generating…"):
            resp = answer(query, course_id=course_id, video_id=video_id)
        st.session_state.last_resp = resp
        if resp["chunks"]:  # jump the player to the most relevant moment
            top = resp["chunks"][0]
            st.session_state.play_video, st.session_state.seek = top.video_id, int(top.start)
        st.rerun()

    resp = st.session_state.last_resp
    if resp:
        st.markdown(f"**Answer**\n\n{resp['answer']}")

        cites = list(dict.fromkeys(re.findall(r"\d{1,2}:\d{2}", resp["answer"])))[:6]
        if cites:
            st.caption("Jump to a cited moment (in the current video):")
            for col, mmss in zip(st.columns(len(cites)), cites):
                if col.button(f"▶ {mmss}", key=f"cite_{mmss}"):
                    st.session_state.seek = to_seconds(mmss)
                    st.rerun()

        with st.expander(f"sources ({len(resp['chunks'])}) — all fed to the LLM"):
            for i, c in enumerate(resp["chunks"]):
                a, b = st.columns([2, 5])
                if a.button(f"▶ {format_timestamp(c.start)}", key=f"src_{i}"):
                    play(c.video_id, c.start)
                b.write(f"`{c.video_id}` · {c.text}" if video_id is None else c.text)

        if resp["frames"]:
            st.caption("Related keyframes (CLIP cross-modal):")
            frames = resp["frames"][:4]
            for j, (col, f) in enumerate(zip(st.columns(len(frames)), frames)):
                if Path(f.frame_path).exists():
                    if col.button(f"▶ {format_timestamp(f.timestamp)}", key=f"kf_{j}"):
                        play(f.video_id, f.timestamp)
                    col.image(f.frame_path)
