from flask import Flask, render_template, request
from PIL import Image
import os
import datetime

# Try to import ultralytics (YOLO)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

app = Flask(__name__)

# Config
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
MODEL_PATH = os.environ.get("MODEL_PATH", "mushroom_model.pt")

# Load model if available
model = None
if YOLO is not None and os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        model = None

# Price table for 8 mushrooms (₹ per kg)
mushroom_prices = {
    "Button": {
        "Jan": "160–200", "Feb": "150–190", "Mar": "140–180", "Apr": "130–170",
        "May": "130–160", "Jun": "140–170", "Jul": "150–180", "Aug": "160–200",
        "Sep": "170–210", "Oct": "180–220", "Nov": "170–210", "Dec": "160–200"
    },
    "Oyster": {
        "Jan": "200–250", "Feb": "180–230", "Mar": "170–220", "Apr": "160–210",
        "May": "150–200", "Jun": "160–210", "Jul": "170–220", "Aug": "180–240",
        "Sep": "190–250", "Oct": "200–260", "Nov": "190–250", "Dec": "180–240"
    },
    "Shiitake": {
        "Jan": "750–900", "Feb": "720–880", "Mar": "700–850", "Apr": "680–820",
        "May": "650–800", "Jun": "670–820", "Jul": "700–850", "Aug": "720–870",
        "Sep": "740–880", "Oct": "760–900", "Nov": "740–880", "Dec": "730–870"
    },
    "Portobello": {
        "Jan": "280–350", "Feb": "300–370", "Mar": "300–380", "Apr": "310–390",
        "May": "320–400", "Jun": "330–420", "Jul": "340–430", "Aug": "350–440",
        "Sep": "360–450", "Oct": "370–460", "Nov": "360–440", "Dec": "340–420"
    },
    "Enoki": {
        "Jan": "950–1200", "Feb": "950–1150", "Mar": "930–1100", "Apr": "900–1050",
        "May": "900–1050", "Jun": "920–1100", "Jul": "950–1150", "Aug": "950–1200",
        "Sep": "960–1200", "Oct": "980–1250", "Nov": "960–1200", "Dec": "950–1200"
    },
    "Morel": {
        "Jan": "16000–20000", "Feb": "15500–19000", "Mar": "15000–18500", "Apr": "14500–18000",
        "May": "14000–17500", "Jun": "14500–18000", "Jul": "15000–18500", "Aug": "15500–19000",
        "Sep": "16000–20000", "Oct": "16500–20500", "Nov": "16000–20000", "Dec": "15500–19500"
    },
    "Paddy Straw": {
        "Jan": "130–170", "Feb": "140–180", "Mar": "150–200", "Apr": "150–200",
        "May": "160–220", "Jun": "170–230", "Jul": "180–250", "Aug": "200–270",
        "Sep": "190–260", "Oct": "170–230", "Nov": "150–200", "Dec": "140–180"
    },
    "Cremini": {
        "Jan": "240–320", "Feb": "250–330", "Mar": "250–340", "Apr": "260–350",
        "May": "260–360", "Jun": "270–370", "Jul": "280–380", "Aug": "280–390",
        "Sep": "290–400", "Oct": "300–410", "Nov": "280–390", "Dec": "260–370"
    }
}

# Map possible labels from model to our canonical 8 names
canonical_map = {
    "button": "Button",
    "white button": "Button",
    "agaricus bisporus": "Button",
    "cremini": "Cremini",
    "crimini": "Cremini",
    "baby bella": "Cremini",
    "portobello": "Portobello",
    "portabella": "Portobello",
    "oyster": "Oyster",
    "pleurotus": "Oyster",
    "shiitake": "Shiitake",
    "lentinula edodes": "Shiitake",
    "enoki": "Enoki",
    "enokitake": "Enoki",
    "paddy straw": "Paddy Straw",
    "paddy-straw": "Paddy Straw",
    "volvariella volvacea": "Paddy Straw",
    "morel": "Morel",
    "gucchi": "Morel",
}

def canonicalize(label: str) -> str:
    if not label:
        return None
    key = label.strip().lower()
    return canonical_map.get(key, label.strip().title())

def predict_label(image_path: str):
    """Return (label, error_message)."""
    if model is None:
        if YOLO is None:
            return None, "Ultralytics not installed. Run: pip install ultralytics"
        if not os.path.exists(MODEL_PATH):
            return None, f"Model not found. Place your trained weights at: {MODEL_PATH}"
        return None, "Model could not be loaded. Check your .pt file."
    try:
        results = model(image_path)
        r = results[0]

        label = None
        # Classification mode
        if hasattr(r, "probs") and r.probs is not None:
            top1 = int(r.probs.top1)
            names = getattr(r, "names", {})
            label = names.get(top1, None)
        # Detection mode (pick highest confidence box)
        if label is None and hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            import torch
            confs = r.boxes.conf
            idx = int(torch.argmax(confs).item()) if hasattr(confs, "argmax") else int(confs.argmax())
            cls_id = int(r.boxes.cls[idx].item())
            names = getattr(r, "names", {})
            label = names.get(cls_id, None)

        if not label:
            return None, "No mushroom detected with high confidence."
        return canonicalize(label), None
    except Exception as e:
        return None, f"Prediction error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    context = {"image": None, "result": None, "price": None, "price_month": None, "error": None, "model_loaded": model is not None}
    if request.method == "POST":
        if "file" not in request.files:
            context["error"] = "No file uploaded!"
            return render_template("index.html", **context)

        file = request.files["file"]
        if file.filename == "":
            context["error"] = "No file selected!"
            return render_template("index.html", **context)

        # Save file
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        context["image"] = filename

        # Predict
        label, err = predict_label(filepath)
        if err:
            context["error"] = err
            return render_template("index.html", **context)

        context["result"] = label

        # Month price
        month = datetime.datetime.now().strftime("%b")  # e.g., 'Sep'
        context["price_month"] = month
        context["price"] = mushroom_prices.get(label, {}).get(month, "Price not available")

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)

