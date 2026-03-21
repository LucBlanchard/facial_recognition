"""
SENTINEL — Face Match System
Place A: reference photo  |  Place B: image / video / webcam

API endpoints used:
  POST /compare_faces
    files: {"reference": <jpeg>, "scene": <jpeg>}
    returns: {"match": bool, "similarity": float, "distance": float, "error": str|null}

  POST /recognize_names   (optional — only used if you also want the name)
    files: {"image": <jpeg>}
    returns: {"names": [...]}

The person in Place A does NOT need to be in the database.
Direct embedding comparison is done via /compare_faces.
"""

import streamlit as st
from PIL import Image
import tempfile
import cv2
import requests
from io import BytesIO
import numpy as np
from datetime import datetime
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Get API endpoint from environment variable or use default
api_endpoint=os.getenv("API_URL","http://localhost:5000")
API_URL              =api_endpoint.rstrip('/')

COMPARE_ENDPOINT     = f"{API_URL}/compare_faces"    # the new endpoint
RECOGNIZE_ENDPOINT   = f"{API_URL}/recognize_names"  # optional name lookup
FRAME_INTERVAL       = 5      # scan every Nth frame (video / webcam)
# NOTE: the actual match decision is made by the API (Euclidean distance < 1.0
# on L2-normalised embeddings, equivalent to cosine similarity > 0.5 / 75%).
# This constant is only used for display purposes in the result banner.
SIMILARITY_THRESHOLD = 0.75

st.set_page_config(
    page_title="SENTINEL — Face Match",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("running",           False)
_init("frame_count",       0)
_init("frames_scanned",    0)
_init("match_found",       False)
_init("best_similarity",   0.0)
_init("match_ts",          "—")
_init("event_log",         [])
_init("video_path",        None)
_init("vid_name_loaded",   None)
_init("ref_image_bytes",   None)   # Place A photo — persisted across reruns
_init("scene_image_bytes", None)   # Place B image — persisted across reruns


# ─────────────────────────────────────────────
#  API HELPERS
# ─────────────────────────────────────────────
def api_compare(ref_bytes: bytes, scene_bgr: np.ndarray) -> dict:
    """
    POST reference image + scene frame to /compare_faces.
    Returns dict: {match, similarity, distance, error}
    Falls back to {match: False, similarity: 0.0} on any error.
    """
    _fallback = {"match": False, "similarity": 0.0, "distance": 999.0, "error": None}
    _, buf = cv2.imencode(".jpg", scene_bgr)
    try:
        r = requests.post(
            COMPARE_ENDPOINT,
            files={
                "reference": ("reference.jpg", ref_bytes,     "image/jpeg"),
                "scene":     ("scene.jpg",     buf.tobytes(), "image/jpeg"),
            },
            timeout=10,
        )
        if r.ok:
            data = r.json()
            # Always show raw API response for debugging
            with st.expander("🔍 API debug — last response", expanded=False):
                st.json(data)
            if data.get("error"):
                st.warning(f"API face detection issue: {data['error']}")
            result = {
                "match":      bool(data.get("match", False)),
                "similarity": float(data.get("similarity", 0.0)),
                "distance":   float(data.get("distance", 999.0)),
                "error":      data.get("error"),
            }
            return result
        else:
            err = f"API returned HTTP {r.status_code}: {r.text[:200]}"
            st.error(err)
            _fallback["error"] = err
    except requests.exceptions.ConnectionError:
        err = "Cannot reach API at localhost:5000 — is the Flask server running?"
        st.error(err)
        _fallback["error"] = err
    except Exception as e:
        err = f"API error: {e}"
        st.error(err)
        _fallback["error"] = err
    return _fallback


def _log_event(src: str, frame_num: int, result: dict, target_label: str):
    """Append one entry to the event log in session state."""
    st.session_state["event_log"].append({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "match":      result["match"],
        "similarity": result["similarity"],
        "distance":   result["distance"],
        "label":      target_label,
        "src":        src,
        "frame":      frame_num,
    })


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title("SENTINEL — Face Match System")
st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.info(
    "**How it works:** Upload any photo in Place A (no database required). "
    "Then upload an image, video, or use your webcam in Place B. "
    "The system will tell you if the same person appears."
)
st.divider()


# ─────────────────────────────────────────────
#  TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────
col_a, col_b = st.columns(2, gap="large")


# ══════════════════════════════════════════════
#  PLACE A — reference photo
# ══════════════════════════════════════════════
with col_a:
    st.subheader("Place A — Reference Subject")
    st.caption("Upload any photo of the person you want to find. No database required.")

    ref_upload = st.file_uploader(
        "Reference photo",
        type=["jpg", "jpeg", "png"],
        key="ref_upload",
        label_visibility="collapsed",
    )

    # Save bytes on new upload; reset dependent state
    if ref_upload is not None:
        new_bytes = ref_upload.read()
        if new_bytes != st.session_state["ref_image_bytes"]:
            st.session_state["ref_image_bytes"]  = new_bytes
            st.session_state["match_found"]      = False
            st.session_state["best_similarity"]  = 0.0
            st.session_state["frames_scanned"]   = 0
            st.session_state["match_ts"]         = "—"
            st.session_state["event_log"]        = []

    # Render from session state (survives reruns / button clicks)
    if st.session_state["ref_image_bytes"]:
        pil_ref = Image.open(BytesIO(st.session_state["ref_image_bytes"]))
        st.image(pil_ref, caption="Reference photo loaded", width="stretch")
        st.success("✅ Reference loaded — ready to compare")

        if st.button("Clear Reference", key="clear_ref", width="stretch"):
            st.session_state.update({
                "ref_image_bytes": None, "match_found": False,
                "best_similarity": 0.0,  "frames_scanned": 0,
                "match_ts": "—",         "event_log": [],
                "ref_image_bytes": None,
            })
            st.rerun()
    else:
        st.info("Upload a reference photo above to get started.")


# ══════════════════════════════════════════════
#  PLACE B — scene
# ══════════════════════════════════════════════
with col_b:
    st.subheader("Place B — Scene to Search")

    ref_ready = bool(st.session_state["ref_image_bytes"])
    if ref_ready:
        st.success("Reference loaded — select a source below and compare.")
    else:
        st.warning("Upload a reference photo in Place A first.")

    source = st.radio(
        "Input source",
        ["Image", "Video File", "Webcam"],
        horizontal=True,
        key="scene_source",
    )

    # Frame display FIRST — anchored before any controls so rerun keeps it in place
    scene_frame_slot = st.empty()
    scene_status     = st.empty()

    # ── IMAGE ──────────────────────────────────
    if source == "Image":
        scene_upload = st.file_uploader(
            "Scene image",
            type=["jpg", "jpeg", "png"],
            key="scene_img",
            label_visibility="collapsed",
        )

        # Save scene bytes to session state on new upload
        if scene_upload is not None:
            new_scene = scene_upload.read()
            if new_scene != st.session_state.get("scene_image_bytes"):
                st.session_state["scene_image_bytes"] = new_scene
                st.session_state["match_found"]       = False
                st.session_state["best_similarity"]   = 0.0
                st.session_state["frames_scanned"]    = 0
                st.session_state["match_ts"]          = "—"

        # Render from session state — survives button-click reruns
        if st.session_state["scene_image_bytes"]:
            pil_scene = Image.open(BytesIO(st.session_state["scene_image_bytes"]))
            scene_frame_slot.image(pil_scene, caption="Scene image", width="stretch")

            if st.button("Compare Now", key="cmp_img", use_container_width=True):
                if not ref_ready:
                    scene_status.error("Upload a reference photo in Place A first.")
                else:
                    with st.spinner("Comparing faces…"):
                        bgr    = cv2.cvtColor(
                            np.array(Image.open(BytesIO(st.session_state["scene_image_bytes"]))),
                            cv2.COLOR_RGB2BGR,
                        )
                        result = api_compare(st.session_state["ref_image_bytes"], bgr)

                        st.session_state["match_found"]     = result["match"]
                        st.session_state["best_similarity"] = result["similarity"]
                        st.session_state["frames_scanned"]  = 1
                        st.session_state["match_ts"]        = datetime.now().strftime("%H:%M:%S")
                        _log_event("IMAGE", 1, result, "reference")
                    st.rerun()
        else:
            scene_frame_slot.info("Upload a scene image above.")

    # ── VIDEO ──────────────────────────────────
    elif source == "Video File":
        vid_upload = st.file_uploader(
            "Surveillance footage",
            type=["mp4", "avi", "mov", "mkv"],
            key="scene_vid",
            label_visibility="collapsed",
        )

        if vid_upload is not None:
            if st.session_state.get("vid_name_loaded") != vid_upload.name:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vid_upload.read())
                tmp.close()
                st.session_state.update({
                    "video_path": tmp.name, "vid_name_loaded": vid_upload.name,
                    "frame_count": 0, "frames_scanned": 0,
                    "running": False, "match_found": False,
                    "best_similarity": 0.0,
                })

        c1, c2, c3 = st.columns(3)

        if c1.button("Start Scan", key="vid_start", use_container_width=True):
            if not ref_ready:
                scene_status.error("Upload a reference photo in Place A first.")
            elif not st.session_state["video_path"]:
                scene_status.error("Upload video footage first.")
            else:
                st.session_state.update({
                    "running": True, "frame_count": 0,
                    "frames_scanned": 0, "match_found": False,
                    "best_similarity": 0.0,
                })
                st.rerun()

        if c2.button("Stop", key="vid_stop", use_container_width=True):
            st.session_state["running"] = False
            st.rerun()

        if c3.button("Reset", key="vid_reset", use_container_width=True):
            st.session_state.update({
                "running": False, "frame_count": 0, "frames_scanned": 0,
                "match_found": False, "best_similarity": 0.0,
                "match_ts": "—", "event_log": [],
                "video_path": None, "vid_name_loaded": None,
            })
            st.rerun()

        # Non-blocking batch processing
        if st.session_state["running"] and st.session_state["video_path"]:
            cap          = cv2.VideoCapture(st.session_state["video_path"])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["frame_count"])

            BATCH     = 10
            ref_bytes = st.session_state["ref_image_bytes"]
            last_rgb  = None

            for _ in range(BATCH):
                ret, frame = cap.read()
                if not ret:
                    st.session_state["running"] = False
                    break

                fc   = st.session_state["frame_count"]
                h, w = frame.shape[:2]

                # Similarity overlay (always shown)
                best_sim_pct = int(st.session_state["best_similarity"] * 100)
                cv2.putText(frame, f"FRAME {fc:06d}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
                cv2.putText(frame, f"BEST MATCH: {best_sim_pct}%",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)

                if fc % FRAME_INTERVAL == 0 and ref_bytes:
                    result = api_compare(ref_bytes, frame)
                    st.session_state["frames_scanned"] += 1

                    if result["similarity"] > st.session_state["best_similarity"]:
                        st.session_state["best_similarity"] = result["similarity"]

                    if result["match"]:
                        st.session_state["match_found"] = True
                        st.session_state["match_ts"]    = datetime.now().strftime("%H:%M:%S")
                        _log_event("VIDEO", fc, result, "reference")
                        cv2.rectangle(frame, (2, 2), (w-2, h-2), (0, 255, 100), 3)
                        cv2.putText(
                            frame,
                            f"MATCH  {int(result['similarity']*100)}%",
                            (w//2 - 80, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2,
                        )

                # Display every frame immediately so the video plays smoothly
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pct = min(100, int((fc + 1) / max(total_frames, 1) * 100))
                scene_frame_slot.image(frame_rgb, width="stretch")
                scene_status.caption(
                    f"Progress: {pct}%  ·  Frame {fc} / {total_frames}  ·  "
                    f"Best similarity: {int(st.session_state['best_similarity']*100)}%"
                )
                st.session_state["frame_count"] += 1

            cap.release()

            if st.session_state["running"]:
                st.rerun()

        elif not st.session_state["video_path"]:
            scene_frame_slot.info("Upload a video file above.")

    # ── WEBCAM ─────────────────────────────────
    elif source == "Webcam":
        wc1, wc2, wc3 = st.columns(3)

        if wc1.button("Start Stream", key="wc_start", use_container_width=True):
            if not ref_ready:
                scene_status.error("Upload a reference photo in Place A first.")
            else:
                st.session_state.update({
                    "running": True, "frame_count": 0,
                    "frames_scanned": 0, "match_found": False,
                    "best_similarity": 0.0,
                })
                st.rerun()

        if wc2.button("Stop", key="wc_stop", use_container_width=True):
            st.session_state["running"] = False
            st.rerun()

        if wc3.button("Reset", key="wc_reset", use_container_width=True):
            st.session_state.update({
                "running": False, "frame_count": 0, "frames_scanned": 0,
                "match_found": False, "best_similarity": 0.0,
                "match_ts": "—", "event_log": [],
            })
            st.rerun()

        if st.session_state["running"]:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                scene_status.error("Webcam unavailable — check connection.")
                st.session_state["running"] = False
            else:
                last_frame = None
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret:
                        last_frame = frame.copy()
                cap.release()

                if last_frame is not None:
                    fc        = st.session_state["frame_count"]
                    ref_bytes = st.session_state["ref_image_bytes"]
                    h, w      = last_frame.shape[:2]

                    best_sim_pct = int(st.session_state["best_similarity"] * 100)
                    cv2.putText(last_frame, f"LIVE  FRAME {fc:06d}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
                    cv2.putText(last_frame, f"BEST MATCH: {best_sim_pct}%",
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)

                    if fc % FRAME_INTERVAL == 0 and ref_bytes:
                        result = api_compare(ref_bytes, last_frame)
                        st.session_state["frames_scanned"] += 1

                        if result["similarity"] > st.session_state["best_similarity"]:
                            st.session_state["best_similarity"] = result["similarity"]

                        if result["match"]:
                            st.session_state["match_found"] = True
                            st.session_state["match_ts"]    = datetime.now().strftime("%H:%M:%S")
                            _log_event("WEBCAM", fc, result, "reference")
                            cv2.rectangle(last_frame, (2, 2), (w-2, h-2), (0, 255, 100), 3)
                            cv2.putText(
                                last_frame,
                                f"MATCH  {int(result['similarity']*100)}%",
                                (w//2 - 80, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2,
                            )

                    scene_frame_slot.image(
                        cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB), width="stretch"
                    )
                    scene_status.caption(
                        f"Live · Frame {fc}  ·  "
                        f"Best similarity: {int(st.session_state['best_similarity']*100)}%"
                    )
                    st.session_state["frame_count"] += 1

            if st.session_state["running"]:
                st.rerun()
        else:
            scene_frame_slot.info("Press Start Stream to activate webcam.")


# ─────────────────────────────────────────────
#  RESULT BANNER
# ─────────────────────────────────────────────
st.divider()
st.subheader("Result")

ref_ready      = bool(st.session_state["ref_image_bytes"])
match_found    = st.session_state["match_found"]
best_sim       = st.session_state["best_similarity"]
frames_scanned = st.session_state["frames_scanned"]
match_ts       = st.session_state["match_ts"]
is_running     = st.session_state["running"]
sim_pct        = int(best_sim * 100)

if not ref_ready:
    st.info("Upload a reference photo in Place A to begin.")
elif match_found:
    st.success(
        f"**MATCH CONFIRMED** — The person from Place A was detected in the scene  \n"
        f"Best similarity: **{sim_pct}%** · Detected at: {match_ts} · "
        f"Frames analysed: {frames_scanned}"
    )
elif is_running:
    st.info(
        f"Scanning…  ·  Frames analysed: {frames_scanned}  ·  "
        f"Best similarity so far: {sim_pct}%"
    )
elif frames_scanned > 0:
    st.error(
        f"**No match detected** — The person from Place A was not found in the scene  \n"
        f"Best similarity seen: **{sim_pct}%**  ·  Frames analysed: {frames_scanned}  \n"
        f"If you expected a match, check the **API debug** expander above for details."
    )
else:
    st.info("Ready — select a source in Place B and start scanning.")


# ─────────────────────────────────────────────
#  STATS ROW
# ─────────────────────────────────────────────
st.divider()
s1, s2, s3, s4 = st.columns(4)
s1.metric("Frames Analysed", frames_scanned)
s2.metric("Best Similarity", f"{sim_pct}%" if frames_scanned > 0 else "—")
s3.metric("Match Found", "YES ✅" if match_found else ("…" if is_running else ("NO ❌" if frames_scanned > 0 else "—")))
s4.metric("Detected At", match_ts if match_found else "—")


# ─────────────────────────────────────────────
#  EVENT LOG  (only match events)
# ─────────────────────────────────────────────
event_log = st.session_state["event_log"]
if event_log:
    st.divider()
    st.subheader("Match Events")
    rows = []
    for entry in reversed(event_log[-20:]):
        rows.append({
            "Time":       entry["time"],
            "Source":     entry["src"],
            "Frame":      entry["frame"],
            "Similarity": f"{int(entry['similarity']*100)}%",
            "Distance":   round(entry["distance"], 3),
            "Match":      "YES" if entry["match"] else "NO",
        })
    st.dataframe(rows, use_container_width=True)