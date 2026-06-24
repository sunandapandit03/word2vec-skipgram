# Bridge Failure Prediction — Your ML Part
==========================================

## Your 5 files and what each one does

| File | Your job | One-line summary |
|---|---|---|
| `model.py` | Define the AI brain | EfficientNet-B3 CNN with 4 output heads |
| `dataset.py` | Load training data | Reads MySQL → preprocesses images |
| `train.py` | Train the model | Teaches the CNN from your bridge images |
| `api.py` | Serve predictions | FastAPI server that Node backend calls |
| `Dockerfile` | Package everything | Runs identically on any machine |

---

## Step 1 — Set up your project folder

Your folder should look like this:
```
your-project/
├── model.py
├── dataset.py
├── train.py
├── api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── backend.js              ← backend person's file
├── package.json            ← backend person's file
├── bridge_db_bridge_images.sql  ← from repo
├── .env                    ← you create this (see Step 2)
└── archive/                ← your Kaggle dataset goes here
    ├── deck/
    │   ├── Cracked/
    │   └── Non-Cracked/
    ├── pavement/
    │   ├── Cracked/
    │   └── Non-Cracked/
    └── wall/
        ├── Cracked/
        └── Non-Cracked/
```

---

## Step 2 — Create your .env file

Create a file named exactly `.env` in your project folder:
```
DB_PASSWORD=your_mysql_password_here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get your Anthropic API key from: https://console.anthropic.com

NEVER commit this file to GitHub. Add `.env` to your `.gitignore`.

---

## Step 3 — Download the dataset

1. Go to: https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification
2. Download and extract
3. Place the images into the `archive/` folder structure shown above

---

## Step 4 — Install Docker

Download Docker Desktop: https://www.docker.com/products/docker-desktop/
Works on Windows, Mac, Linux.

---

## Step 5 — Train the model (first time only)

```bash
# This builds your container and runs training
docker-compose run ml_api python train.py
```

What you'll see:
```
Training on: cpu (or cuda if you have a GPU)
Loaded 40,000 records from MySQL.
Train: 32,000 images | Val: 8,000 images
Model ready. Total params: 12,345,678 | Trainable: 2,345,678
Epoch 01/20 | Train loss: 1.2345  crack_acc: 72.3%  | Val loss: 0.9876  crack_acc: 81.2%
...
Epoch 10/20 | Train loss: 0.3456  crack_acc: 91.5%  | Val loss: 0.2987  crack_acc: 93.1%
Entering Phase 2 — unfreezing backbone for fine-tuning
...
Epoch 20/20 | Train loss: 0.1234  crack_acc: 96.2%  | Val loss: 0.1456  crack_acc: 95.8%
Training complete. Best model saved to: checkpoints/bridge_model.pt
```

The trained model is saved to `checkpoints/bridge_model.pt` on your laptop.

---

## Step 6 — Start everything

```bash
docker-compose up --build
```

This starts:
- MySQL on port 3306
- Your ML API on port 8000
- Node backend on port 3000

---

## Step 7 — Test your ML API directly

Open your browser: http://localhost:8000/docs

This shows the interactive API documentation where you can upload a test image
and see the full JSON response.

Or from the terminal:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/bridge_image.jpg"
```

---

## What the JSON response looks like

```json
{
  "damage_detected": true,
  "damage_confidence": 0.9231,
  "structure_type": "deck",
  "severity_score": 38.2,
  "days_to_failure": 87,
  "risk_level": "high",
  "class_probabilities": {
    "non-cracked": 0.0769,
    "cracked": 0.9231
  },
  "report": {
    "what": "Significant flexural cracking was detected on the bridge deck...",
    "why": "The cracking pattern suggests repeated load cycling combined with...",
    "when": "At the current deterioration rate, critical intervention is needed within 87 days...",
    "next_steps": [
      "Schedule emergency inspection by a certified structural engineer",
      "Apply epoxy injection to seal active cracks immediately",
      "Install strain gauges for real-time structural monitoring",
      "Review load limits and restrict heavy vehicle access"
    ]
  }
}
```

---

## Coordination with your backend person

She needs two things from you:
1. The ML API URL: `http://ml_api:8000` (inside Docker) or `http://localhost:8000` (local testing)
2. The field name for the image: `"file"` (in her `form.append('file', ...)` call)

The JSON fields her backend saves to MySQL:
- `crack_detected` → `damage_detected`
- `severity` → `severity_score`
- `risk_level` → `risk_level`
- `report` → the full report object (she can stringify it or store sections separately)

---

## If you don't have a GPU (training is slow)

Use Google Colab (free):
1. Upload your files to Google Drive
2. Open Colab, mount your Drive
3. Run train.py — Colab gives you a free GPU
4. Download the trained `bridge_model.pt`
5. Put it in your `checkpoints/` folder
6. Run the API server on your laptop
