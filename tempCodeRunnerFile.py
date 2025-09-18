from flask import Flask, render_template, request
import os
import json
import torch
from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
except:
    YOLO = None

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join("static", "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
MODEL_PATH = os.path.join("model", "mushroom_model.pt")
model = None
if YOLO and os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
        print("✅ YOLO Model loaded successfully!")
    except Exception as e:
        print("❌ Model loading error:", e)
        model = None
else:
    print("❌ Model file not found or YOLO not installed.")

# Load mushroom info JSON
INFO_PATH = "mushroom_info.json"
mushroom_data = {}
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        # If list → convert to dict
        if isinstance(data, list):
            mushroom_data = {m["label"]: m for m in data}
        elif isinstance(data, dict):
            mushroom_data = data
else:
    print("❌ mushroom_info.json not found!")


def predict_label(image_path: str):
    """Predict mushroom label using YOLO model."""
    if model is None:
        return None, "Model not loaded."
    try:
        results = model.predict(image_path)
        if not results or len(results) == 0:
            return None, "No prediction results."

        r = results[0]
        label = None
        names = getattr(r, "names", model.names if model else {})

        # Classification
        if hasattr(r, "probs") and r.probs is not None:
            top1 = int(r.probs.top1)
            label = names.get(top1)

        # Detection
        elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf
            idx = int(torch.argmax(confs).item())
            cls_id = int(r.boxes.cls[idx].item())
            label = names.get(cls_id)

        if not label:
            return None, "No mushroom detected."

        return label.lower().replace(" ", "_"), None

    except Exception as e:
        return None, f"Prediction error: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "image": None,
        "name": None,
        "desc": None,
        "cultivation": None,
        "nutrients": [],
        "price": None,
        "price_month": None,
        "error": None,
        "model_loaded": model is not None
    }

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            context["error"] = "No file selected!"
            return render_template("index.html", **context)

        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        context["image"] = f"data/{filename}"

        # Predict
        label, err = predict_label(filepath)
        if err:
            context["error"] = err
            return render_template("index.html", **context)

        # Get selected options
        month = request.form.get("month")
        lang = request.form.get("lang", "en")
        mushroom = mushroom_data.get(label, {})

        if not mushroom:
            context["error"] = f"No data found for label: {label}"
            return render_template("index.html", **context)

        # Language handling
        if lang == "hi":
            context["name"] = mushroom.get("name_hi", mushroom.get("name_en"))
            context["desc"] = mushroom.get("description_hi", mushroom.get("description_en"))
            context["cultivation"] = mushroom.get("cultivation_hi", mushroom.get("cultivation_en"))
        else:
            context["name"] = mushroom.get("name_en", mushroom.get("name_hi"))
            context["desc"] = mushroom.get("description_en", mushroom.get("description_hi"))
            context["cultivation"] = mushroom.get("cultivation_en", mushroom.get("cultivation_hi"))

        # ✅ Nutrients add
        context["nutrients"] = mushroom.get("nutrients", [])

        # Normalize month key
        month_key = month.strip().capitalize() if month else None
        context["price_month"] = month_key
        context["price"] = mushroom.get("price", {}).get(month_key, "N/A")

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5008)
