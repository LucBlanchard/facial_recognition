import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import os

API_URL = "http://localhost:5000"
# Get API endpoint from environment variable or use default
api_endpoint=os.getenv("API_URL","http://localhost:5000")
API_URL=api_endpoint.rstrip('/')

st.set_page_config(page_title="Face Recognition Manager", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0f0f0f;
    color: #e8e6e0;
}

h1, h2, h3 {
    font-family: 'DM Mono', monospace !important;
    letter-spacing: -0.02em;
}

.face-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    transition: border-color 0.2s;
}

.face-card:hover {
    border-color: #3a3a3a;
}

.avatar {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
}

.badge {
    background: #1e2a1e;
    color: #5a9e5a;
    border: 1px solid #2a3d2a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
}

.stat-box {
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.stat-number {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #e8e6e0;
}

.stat-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

div[data-testid="stButton"] > button {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    border: 1px solid #2a2a2a;
    background: #1a1a1a;
    color: #e8e6e0;
    padding: 0.4rem 1rem;
    transition: all 0.15s;
}

div[data-testid="stButton"] > button:hover {
    border-color: #444;
    background: #222;
}

div[data-testid="stTextInput"] input {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    color: #e8e6e0;
    font-family: 'DM Sans', sans-serif;
}

div[data-testid="stFileUploader"] {
    background: #141414;
    border: 1px dashed #2a2a2a;
    border-radius: 10px;
}

.stSuccess {
    background: #1a2a1a;
    border: 1px solid #2a3d2a;
    color: #5a9e5a;
    border-radius: 8px;
}

.stError {
    background: #2a1a1a;
    border: 1px solid #3d2a2a;
    border-radius: 8px;
}

.stWarning {
    background: #2a2210;
    border: 1px solid #3d3010;
    border-radius: 8px;
}

hr {
    border-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)


# ── Avatar colors ────────────────────────────────────────────────
AVATAR_COLORS = [
    ("#1a2a3a", "#4a8fbf"),
    ("#1a2a1a", "#4a9e4a"),
    ("#2a1a2a", "#9e4abf"),
    ("#2a2010", "#c8841a"),
    ("#2a1a1a", "#bf4a4a"),
    ("#102a2a", "#1abfbf"),
]

def initials(name):
    parts = name.strip().split()
    return "".join(p[0].upper() for p in parts[:2])

def avatar_html(name, idx):
    bg, fg = AVATAR_COLORS[idx % len(AVATAR_COLORS)]
    return f"""
    <div style="width:72px;height:72px;border-radius:50%;
                background:{bg};color:{fg};border:1.5px solid {fg}33;
                display:flex;align-items:center;justify-content:center;
                font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:500;
                margin:0 auto 8px;">
        {initials(name)}
    </div>"""


# ── API helpers ──────────────────────────────────────────────────
def api_get_people():
    try:
        r = requests.get(f"{API_URL}/people", timeout=5)
        if r.ok:
            return r.json()
        return {}
    except requests.exceptions.ConnectionError:
        return None

def api_add_person(name, files):
    try:
        file_list = [("images", (f.name, f.getvalue(), f.type)) for f in files]
        r = requests.post(f"{API_URL}/add", data={"name": name}, files=file_list, timeout=30)
        return r.ok, r.json().get("message", "Done")
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach the Flask server."

def api_remove_person(name):
    try:
        r = requests.delete(f"{API_URL}/remove", json={"name": name}, timeout=10)
        return r.ok, r.json().get("message", "Done")
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach the Flask server."


# ── Page header ──────────────────────────────────────────────────
st.markdown("## face_recognition_manager")
st.markdown("<p style='color:#555;font-size:0.85rem;margin-top:-12px;margin-bottom:1.5rem;'>Manage registered identities and embeddings</p>", unsafe_allow_html=True)

# ── Fetch data ───────────────────────────────────────────────────
people_data = api_get_people()

if people_data is None:
    st.error("Cannot reach Flask server at `http://localhost:5000`. Make sure it is running.")
    st.stop()

people = list(people_data.items())  # [(name, count), ...]

# ── Stats row ────────────────────────────────────────────────────
total_images = sum(c for _, c in people)
avg_images = round(total_images / len(people)) if people else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{len(people)}</div>
        <div class="stat-label">People registered</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{total_images}</div>
        <div class="stat-label">Total images</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{avg_images}</div>
        <div class="stat-label">Avg images / person</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Registered faces ─────────────────────────────────────────────
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<p class="section-header">— registered faces</p>', unsafe_allow_html=True)

    search = st.text_input("", placeholder="Search by name…", label_visibility="collapsed")
    filtered = [(n, c) for n, c in people if search.lower() in n.lower()]

    KNOWN_FACES_DIR = "C:/Users/INVICTUS/Documents/POO/Face-Recognition-using-YoloV8-and-FaceNet/known_faces"

    def get_first_image(name):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_dir):
            for f in os.listdir(person_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    return os.path.join(person_dir, f)
        return None

    if not filtered:
        st.markdown("<p style='color:#444;font-size:0.85rem;padding:1rem 0'>No faces registered yet.</p>", unsafe_allow_html=True)
    else:
        cols_per_row = 4
        for row_start in range(0, len(filtered), cols_per_row):
            row_people = filtered[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (i, (name, count)) in zip(cols, enumerate(row_people)):
                with col:
                    img_path = get_first_image(name)
                    if img_path:
                        st.image(Image.open(img_path), use_container_width=True)
                    else:
                        # fallback to initials avatar if no image found
                        idx = row_start + i
                        st.markdown(avatar_html(name, idx), unsafe_allow_html=True)

                    st.markdown(f"<div style='font-size:0.85rem;font-weight:500;text-align:center;color:#e8e6e0;margin-top:6px'>{name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='badge' style='text-align:center;margin:4px auto'>{count} img{'s' if count != 1 else ''}</div>", unsafe_allow_html=True)

                    if st.button("Remove", key=f"remove_{name}", use_container_width=True):
                        ok, msg = api_remove_person(name)
                        if ok:
                            st.success(f"{name} removed.")
                        else:
                            st.error(msg)
                        st.rerun()

# ── Add person ───────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-header">— add a person</p>', unsafe_allow_html=True)

    new_name = st.text_input("Name", placeholder="e.g. OBAME")
    uploaded_files = st.file_uploader(
        "Photos",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Upload one or more face photos for this person."
    )

    if uploaded_files:
        st.markdown(f"<p style='color:#555;font-size:0.78rem'>{len(uploaded_files)} file(s) selected</p>", unsafe_allow_html=True)
        preview_cols = st.columns(min(len(uploaded_files), 4))
        for col, f in zip(preview_cols, uploaded_files[:4]):
            with col:
                img = Image.open(f)
                st.image(img, use_container_width=True)
                f.seek(0)

    if st.button("Save to system", use_container_width=True):
        if not new_name.strip():
            st.warning("Enter a name first.")
        elif not uploaded_files:
            st.warning("Upload at least one photo.")
        else:
            ok, msg = api_add_person(new_name.strip(), uploaded_files)
            if ok:
                st.success(f"{new_name} added. Re-run `generate_face_embeddings.py` to update embeddings.")
                st.rerun()
            else:
                st.error(msg)