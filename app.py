import io, os, csv, uuid
from datetime import datetime
from flask import Flask, request, send_file, jsonify, render_template_string
from werkzeug.utils import secure_filename

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

ALLOWED_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

INDEX_HTML = """
<!doctype html>
<title>Pose Landmarks → CSV</title>
<h1>Upload a video</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=video required>
  <button type=submit>Process</button>
</form>
<p>Returns a CSV with per-frame 33 landmarks × (x,y,z,visibility).</p>
"""

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", time=datetime.utcnow().isoformat()+"Z")

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify(error="No file part"), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify(error="No selected file"), 400
    if not allowed(f.filename):
        return jsonify(error="Unsupported file type"), 400

    filename = secure_filename(f.filename)
    uid = uuid.uuid4().hex
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{filename}")
    f.save(save_path)

    cap = cv2.VideoCapture(save_path)
    if not cap.isOpened():
        return jsonify(error="Failed to open video"), 400

    output = io.StringIO()
    writer = csv.writer(output)

    cols = ["frame"]
    for i in range(33):
        cols += [f"lmk_{i}_x", f"lmk_{i}_y", f"lmk_{i}_z", f"lmk_{i}_v"]
    writer.writerow(cols)

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            row = [frame_idx]
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                for i in range(33):
                    l = lm[i]
                    row += [l.x, l.y, l.z, getattr(l, "visibility", 0.0)]
            else:
                row += [0.0] * (33 * 4)
            writer.writerow(row)
            frame_idx += 1

    cap.release()

    csv_bytes = io.BytesIO(output.getvalue().encode("utf-8"))
    csv_name = f"pose_landmarks_{uid}.csv"
    return send_file(csv_bytes, mimetype="text/csv", as_attachment=True, download_name=csv_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
