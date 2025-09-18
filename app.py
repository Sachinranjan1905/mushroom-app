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

# -------------------- Debug / Paths --------------------
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/data')
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/mushroom_model.pt')
INFO_PATH = 'mushroom_info.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Debug messages
print("🔹 Checking model and JSON files...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at: {MODEL_PATH}")
else:
    print(f"✅ Model file found: {MODEL_PATH}")

if not os.path.exists(INFO_PATH):
    print(f"❌ mushroom_info.json not found at: {INFO_PATH}")
else:
    print(f"✅ mushroom_info.json found: {INFO_PATH}")

# -------------------- Load YOLO Model --------------------
model = None
if YOLO and os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
        model.to("cpu")  # Force CPU to avoid GPU crashes on Render
        print("✅ YOLO Model loaded successfully on CPU!")
    except Exception as e:
        print("❌ Model loading error:", e)


# -------------------- Load JSON --------------------
mushroom_data = {}
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            mushroom_data = {m["label"]: m for m in data}
        elif isinstance(data, dict):
            mushroom_data = data

# -------------------- Prediction Function --------------------
def predict_label(image_path: str):
    if model is None:
        return None, "Model not loaded."
    try:
        results = model.predict(image_path)
        if not results or len(results) == 0:
            return None, "No prediction results."
        r = results[0]
        label = None
        names = getattr(r, "names", model.names if model else {})
        if hasattr(r, "probs") and r.probs is not None:
            top1 = int(r.probs.top1)
            label = names.get(top1)
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

# -------------------- Flask Routes --------------------
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

        # save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        context["image"] = f"data/{filename}"

        # prediction
        label, err = predict_label(filepath)
        if err:
            context["error"] = err
            return render_template("index.html", **context)

        month = request.form.get("month")
        lang = request.form.get("lang", "en")
        mushroom = mushroom_data.get(label, {})

        if not mushroom:
            context["error"] = f"No data found for label: {label}"
            return render_template("index.html", **context)

        # Language-specific content
        if lang == "hi":
            context["name"] = mushroom.get("name_hi", mushroom.get("name_en"))
            context["desc"] = mushroom.get("description_hi", mushroom.get("description_en"))
            context["cultivation"] = mushroom.get("cultivation_hi", mushroom.get("cultivation_en"))
            nutrients = mushroom.get("nutrients_hi", mushroom.get("nutrients_en"))
        else:
            context["name"] = mushroom.get("name_en", mushroom.get("name_hi"))
            context["desc"] = mushroom.get("description_en", mushroom.get("description_hi"))
            context["cultivation"] = mushroom.get("cultivation_en", mushroom.get("cultivation_hi"))
            nutrients = mushroom.get("nutrients_en", mushroom.get("nutrients_hi"))

        # ✅ fix nutrients handling (always list)
        # Nutrients (common field in JSON, no _en/_hi)
        nutrients = mushroom.get("nutrients", [])
        if isinstance(nutrients, list):
            context["nutrients"] = nutrients
        elif isinstance(nutrients, str):
            context["nutrients"] = [nutrients]
        else:
            context["nutrients"] = []

        # Price handling
        month_key = month.strip().capitalize() if month else None
        context["price_month"] = month_key
        context["price"] = mushroom.get("price", {}).get(month_key, "N/A")

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5008)
