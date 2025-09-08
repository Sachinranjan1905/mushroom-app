# Mushroom Classification Web App (Flask)

A simple Flask app that:
- Lets users upload a mushroom image
- Runs a trained YOLO model to classify it
- Shows the **current month's** price range (₹/kg) for 8 mushroom types

## Quick Start

1) **Install Python 3.10+**

2) **Create & activate a virtual environment**

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) **Install dependencies**
```bash
pip install -r requirements.txt
```

> If PyTorch doesn't install automatically on your platform, visit https://pytorch.org/get-started/locally/ and install a CPU wheel compatible with your Python version.

4) **Add your trained model**
- Place your YOLO weights as `mushroom_model.pt` in the project root (or set `MODEL_PATH` env var).

5) **Run the app**
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser.

### Access on your phone (same Wi‑Fi)
The app already runs with `host=0.0.0.0`. Find your PC's local IP (e.g. `192.168.1.10`) and open:
```
http://192.168.1.10:5000
```
on your phone connected to the same network.

## Notes

- The app gracefully shows a message if the model isn't found or Ultralytics isn't installed.
- Month-wise prices are hardcoded for 8 mushrooms: Button, Oyster, Shiitake, Portobello, Enoki, Morel, Paddy Straw, Cremini.
- You can change prices in `app.py` (the `mushroom_prices` dict).

## Deploying (optional)

- **Render / Railway / PythonAnywhere**: Push repo to GitHub and connect.
- Add `web: gunicorn app:app` to a `Procfile` and set `MODEL_PATH` to your weights path.
- For storage of uploads in production, use a temp directory or cloud storage.
# mushroom-app-starter
# mushroom-app-starter
# mushroom-app-starter
