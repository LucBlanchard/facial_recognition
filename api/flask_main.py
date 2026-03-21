import torch
import cv2
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from flask import Flask, request, jsonify, send_file
import pickle
from io import BytesIO
import shutil


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KNOWN_FACES_DIR = "C:/Users/INVICTUS/Documents/POO/Face-Recognition-using-YoloV8-and-FaceNet/known_faces"
EMBEDDINGS_FILE = "known_embeddings.pkl"

try:
    model = YOLO("../detection/weights/best.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

try:
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("MTCNN and InceptionResnetV1 models loaded successfully.")
except Exception as e:
    print(f"Error loading MTCNN/InceptionResnetV1: {e}")
    exit()

def load_known_embeddings():
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            known_embeddings = pickle.load(f)
            print("Known embeddings loaded successfully.")
    except Exception as e:
        known_embeddings = {}
        print(f"Error loading known embeddings: {e}")
    return known_embeddings

known_embeddings = load_known_embeddings()


def compare_embeddings(face_embedding, known_embeddings):
    threshold = 0.9
    min_dist = float('inf')
    match = "Unknown"
    for name, embedding_list in known_embeddings.items():
        for known_embedding in embedding_list:
            dist = np.linalg.norm(np.array(face_embedding) - np.array(known_embedding))
            if dist < min_dist:
                min_dist = dist
                match = name if dist < threshold else "Unknown"
    print(f"Min distance: {min_dist}, Match: {match}")
    return match


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPER — accepts a BGR numpy array (NOT a file path)
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding_from_bgr(bgr_img, label="image"):
    """
    Extract a 512-d face embedding from a BGR numpy array.

    Steps:
      1. Try YOLO → crop the largest face box → MTCNN align → ResNet embed
      2. If YOLO finds nothing → run MTCNN directly on the full image
      3. If MTCNN alignment returns None → resize raw crop and embed anyway

    Returns (embedding: np.ndarray | None, error: str | None)
    """
    full_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    face_rgb_to_embed = None

    # ── Step 1: YOLO crop ────────────────────────────────────────────────
    results    = model(bgr_img)
    yolo_boxes = results[0].boxes

    if len(yolo_boxes) > 0:
        best_box, best_area = None, 0
        for box in yolo_boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)

        x1, y1, x2, y2 = best_box
        crop = bgr_img[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] >= 20 and crop.shape[1] >= 20:
            face_rgb_to_embed = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            print(f"[{label}] YOLO crop {crop.shape}")
        else:
            print(f"[{label}] YOLO crop too small, falling back")
    else:
        print(f"[{label}] YOLO: no detections, falling back to MTCNN on full image")

    # ── Step 2: MTCNN on full image if YOLO failed ───────────────────────
    if face_rgb_to_embed is None:
        try:
            boxes, _ = mtcnn.detect(full_rgb)
            if boxes is not None and len(boxes) > 0:
                best_box, best_area = None, 0
                for box in boxes:
                    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                    x2 = min(full_rgb.shape[1], x2)
                    y2 = min(full_rgb.shape[0], y2)
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_box  = (x1, y1, x2, y2)
                if best_box:
                    x1, y1, x2, y2 = best_box
                    crop = full_rgb[y1:y2, x1:x2]
                    if crop.shape[0] >= 20 and crop.shape[1] >= 20:
                        face_rgb_to_embed = crop
                        print(f"[{label}] MTCNN full-image crop {crop.shape}")
        except Exception as e:
            print(f"[{label}] MTCNN detect failed: {e}")

    # ── Step 3: last resort — use the whole image ────────────────────────
    if face_rgb_to_embed is None:
        face_rgb_to_embed = full_rgb
        print(f"[{label}] using full image as last resort")

    # ── MTCNN align + ResNet embed ───────────────────────────────────────
    try:
        face_tensors = mtcnn(face_rgb_to_embed)
    except RuntimeError as e:
        return None, f"MTCNN failed on {label}: {e}"

    if face_tensors is None:
        # Alignment failed — resize raw crop and embed directly
        try:
            from torchvision import transforms
            resize_tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
            t = resize_tf(face_rgb_to_embed).unsqueeze(0)
            emb = resnet(t).detach().cpu().numpy()[0]
            print(f"[{label}] used resize fallback")
            return emb, None
        except Exception as e2:
            return None, f"All methods failed for {label}: {e2}"

    if face_tensors.dim() == 4:
        face_tensors = face_tensors[0:1]
    else:
        face_tensors = face_tensors.unsqueeze(0)

    emb = resnet(face_tensors).detach().cpu().numpy()[0]
    print(f"[{label}] embedding OK, shape={emb.shape}")
    return emb, None


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — convert a Flask FileStorage → BGR numpy array
# ─────────────────────────────────────────────────────────────────────────────
def filestorage_to_bgr(file_storage):
    """Read a Flask FileStorage (uploaded image) and return a BGR numpy array."""
    pil = Image.open(file_storage.stream).convert("RGB")
    arr = np.array(pil)                          # RGB uint8
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # → BGR


@app.route("/")
def home():
    return "<h3>Face Recognition API</h3>"


@app.route("/people", methods=['GET'])
def get_people():
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            ke = pickle.load(f)
        return jsonify({name: len(embs) for name, embs in ke.items()})
    except FileNotFoundError:
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/add", methods=['POST'])
def add_person():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"message": "Name is required."}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"message": "At least one image is required."}), 400

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    saved = []
    for f in files:
        save_path = os.path.join(person_dir, f.filename)
        f.save(save_path)
        saved.append(save_path)

    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            ke = pickle.load(f)
    except FileNotFoundError:
        ke = {}

    person_embeddings = []
    for img_path in saved:
        img = cv2.imread(img_path)
        if img is None:
            continue
        yolo_results = model(img)
        for box in yolo_results[0].boxes.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if face_rgb.shape[0] < 20 or face_rgb.shape[1] < 20:
                continue
            try:
                face_tensors = mtcnn(face_rgb)
                if face_tensors is None:
                    continue
                emb = resnet(face_tensors).detach().cpu().numpy()[0]
                person_embeddings.append(emb)
            except RuntimeError:
                continue

    if not person_embeddings:
        shutil.rmtree(person_dir)
        return jsonify({"message": f"No faces detected for '{name}'."}), 400

    ke[name] = person_embeddings
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(ke, f)

    return jsonify({"message": f"{name} added with {len(person_embeddings)} embedding(s)."}), 200


@app.route("/remove", methods=['DELETE'])
def remove_person():
    data = request.get_json()
    name = data.get("name", "").strip() if data else ""
    if not name:
        return jsonify({"message": "Name is required."}), 400

    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            ke = pickle.load(f)
    except FileNotFoundError:
        return jsonify({"message": "Embeddings file not found."}), 404

    if name not in ke:
        return jsonify({"message": f"'{name}' not found."}), 404

    del ke[name]
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(ke, f)

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir)

    return jsonify({"message": f"{name} removed successfully."}), 200


@app.route("/recognize", methods=['POST'])
def process_frame():
    frame = filestorage_to_bgr(request.files['image'])

    results    = model(frame)
    yolo_boxes = results[0].boxes
    print(f"Faces detected: {len(yolo_boxes)}")

    faces, face_coords = [], []
    for box in yolo_boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        faces.append(face)
        face_coords.append((x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, face in enumerate(faces):
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if face_rgb.shape[0] < 20 or face_rgb.shape[1] < 20:
            continue
        try:
            mtcnn_boxes, _ = mtcnn.detect(face_rgb)
        except RuntimeError:
            continue
        if mtcnn_boxes is not None:
            try:
                face_tensors = mtcnn(face_rgb)
            except RuntimeError:
                continue
            if face_tensors is None:
                continue
            face_emb = resnet(face_tensors).detach().cpu().numpy()[0]
            match    = compare_embeddings(face_emb, known_embeddings)
            x1, y1, x2, y2 = face_coords[i]
            cv2.putText(frame, match, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')


@app.route("/recognize_names", methods=['POST'])
def recognize_names():
    frame = filestorage_to_bgr(request.files['image'])

    results    = model(frame)
    yolo_boxes = results[0].boxes
    print(f"Faces detected: {len(yolo_boxes)}")

    names_found = []
    for box in yolo_boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if face_rgb.shape[0] < 20 or face_rgb.shape[1] < 20:
            continue
        try:
            mtcnn_boxes, _ = mtcnn.detect(face_rgb)
        except RuntimeError:
            continue
        if mtcnn_boxes is None:
            continue
        try:
            face_tensors = mtcnn(face_rgb)
        except RuntimeError:
            continue
        if face_tensors is None:
            continue
        face_emb = resnet(face_tensors).detach().cpu().numpy()[0]
        match    = compare_embeddings(face_emb, known_embeddings)
        if match != "Unknown":
            names_found.append(match)
            print(f"Recognized: {match}")

    return jsonify({"names": names_found}), 200


@app.route("/compare_faces", methods=["POST"])
def compare_faces():
    """
    Compare two faces directly — no database required.

    POST with:
        files["reference"]  →  Place A photo
        files["scene"]      →  Place B frame / photo

    Returns:
        {"match": bool, "similarity": float 0-1, "distance": float, "error": str|null}
    """
    if "reference" not in request.files or "scene" not in request.files:
        return jsonify({
            "error": "Both 'reference' and 'scene' image fields are required.",
            "match": False, "similarity": 0.0, "distance": 999.0
        }), 400

    # ── Convert FileStorage → BGR numpy array (the correct way) ──────────
    ref_bgr   = filestorage_to_bgr(request.files["reference"])
    scene_bgr = filestorage_to_bgr(request.files["scene"])

    # ── Extract embeddings ────────────────────────────────────────────────
    ref_emb,   ref_err   = get_embedding_from_bgr(ref_bgr,   label="reference")
    scene_emb, scene_err = get_embedding_from_bgr(scene_bgr, label="scene")

    if ref_emb is None:
        print(f"[compare_faces] reference failed: {ref_err}")
        return jsonify({"match": False, "similarity": 0.0,
                        "distance": 999.0, "error": ref_err}), 200

    if scene_emb is None:
        print(f"[compare_faces] scene failed: {scene_err}")
        return jsonify({"match": False, "similarity": 0.0,
                        "distance": 999.0, "error": scene_err}), 200

    # ── L2-normalise then compare ─────────────────────────────────────────
    ref_n   = ref_emb   / (np.linalg.norm(ref_emb)   + 1e-10)
    scene_n = scene_emb / (np.linalg.norm(scene_emb) + 1e-10)

    distance   = float(np.linalg.norm(ref_n - scene_n))   # 0 = identical, 2 = opposite
    cosine     = float(np.dot(ref_n, scene_n))             # 1 = identical, -1 = opposite
    similarity = (cosine + 1.0) / 2.0                     # mapped to [0, 1]

    THRESHOLD = 1.0   # distance < 1.0  ↔  similarity > 75%
    match     = distance < THRESHOLD

    print(
        f"[compare_faces] dist={distance:.4f}  "
        f"cosine={cosine:.4f}  sim={similarity:.4f} ({int(similarity*100)}%)  match={match}"
    )

    return jsonify({
        "match":      match,
        "similarity": round(similarity, 4),
        "distance":   round(distance,   4),
        "error":      None,
    }), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)