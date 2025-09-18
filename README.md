# 🍄 Mushroom Classification Web App (Flask)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-F7931E?style=for-the-badge&logo=machinelearning&logoColor=white)
![GitHub stars](https://img.shields.io/github/stars/Sachinranjan1905/mushroom-app?style=social)
![GitHub forks](https://img.shields.io/github/forks/Sachinranjan1905/mushroom-app?style=social)
![License](https://img.shields.io/badge/License-MIT-green)

A **Flask web app** that:  
- Lets users **upload a mushroom image**  
- Runs a trained **YOLOv12s model** to classify it  
- Shows the **current month's price range (₹/kg)** for 8 mushroom types

---

## 🔹 Demo Screenshots / GIF

> Replace the placeholders with actual screenshots or GIFs from your app

![App Home Page](/Users/sachinranjan/mushroom-app-starter/Screenshot 2025-09-18 at 11.24.58 AM.png)  
![Mushroom Detection](screenshots/detection.gif)  

---

## 🔹 Quick Start

### 1️⃣ Install Python 3.10+

### 2️⃣ Create & Activate a Virtual Environment

**Windows (PowerShell)**
# 1️⃣ Create virtual environment
python -m venv .venv

# 2️⃣ Activate it (PowerShell)
.venv\Scripts\Activate.ps1



2) **macOS / Linux**
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
Open http://127.0.0.1:5008 in your browser.

### Access on your phone (same Wi‑Fi)
The app already runs with `host=0.0.0.0`. Find your PC's local IP (e.g. `192.168.1.10`) and open:
```
http://192.168.1.10:5008
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
## Project Structure
 /mushroom-web-app
 ├── app.py                 # Main Flask app
 ├── static/                # CSS, JS, images
 ├── templates/             # HTML templates
 ├── mushroom_model.pt      # YOLOv12s weights
 ├── screenshots/           # Demo screenshots / GIFs
 ├── requirements.txt       # Dependencies
 └── README.md              # Project documentation
## Author
Sachin Ranjan
B.Tech CSE '28, Quantum University
GitHub: Sachinranjan1905
5⭐ C++ @ HackerRank

# mushroom-app-starter

