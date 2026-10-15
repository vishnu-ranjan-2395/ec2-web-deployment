
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(
    title="ML Flower Predictor",
    description="Iris flower prediction using Scikit-learn",
    version="1.0.0",
)

model = None


def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model is missing. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    load_model()


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>ML Flower Predictor</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background: #eef3fb;
          margin: 0;
          padding: 30px 16px;
          color: #172033;
        }
        main {
          max-width: 560px;
          margin: 30px auto;
          background: white;
          padding: 28px;
          border-radius: 18px;
          box-shadow: 0 8px 30px #17203318;
        }
        h1 { color: #174ea6; }
        .fields {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 14px;
        }
        label { display: block; font-size: 14px; }
        input {
          box-sizing: border-box;
          width: 100%;
          padding: 12px;
          margin-top: 7px;
          border: 1px solid #ccd4e0;
          border-radius: 8px;
        }
        button {
          width: 100%;
          margin-top: 22px;
          padding: 13px;
          border: 0;
          border-radius: 8px;
          background: #174ea6;
          color: white;
          font-size: 16px;
          cursor: pointer;
        }
        #result {
          margin-top: 18px;
          padding: 14px;
          border-radius: 8px;
          background: #eef3fb;
          white-space: pre-wrap;
        }
        @media (max-width: 420px) {
          .fields { grid-template-columns: 1fr; }
        }
      </style>
    </head>
    <body>
      <main>
        <h1>🌸 ML Flower Predictor</h1>
        <p>Enter flower measurements in centimetres.</p>
        <form id="predictForm">
          <div class="fields">
            <label>Sepal length
              <input name="sepal_length" type="number"
                step="0.1" min="0.1" max="10" value="5.1" required>
            </label>
            <label>Sepal width
              <input name="sepal_width" type="number"
                step="0.1" min="0.1" max="10" value="3.5" required>
            </label>
            <label>Petal length
              <input name="petal_length" type="number"
                step="0.1" min="0.1" max="10" value="1.4" required>
            </label>
            <label>Petal width
              <input name="petal_width" type="number"
                step="0.1" min="0.1" max="10" value="0.2" required>
            </label>
          </div>
          <button type="submit">Predict Flower</button>
        </form>
        <div id="result" role="status">Your prediction will appear here.</div>
      </main>
      <script>
        document.getElementById("predictForm").addEventListener(
          "submit", async (event) => {
            event.preventDefault();
            const result = document.getElementById("result");
            result.textContent = "Predicting...";
            const data = Object.fromEntries(new FormData(event.target));
            for (const key in data) data[key] = Number(data[key]);
            try {
              const response = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(data)
              });
              const answer = await response.json();
              if (!response.ok) throw new Error(answer.detail || "Prediction failed");
              result.textContent =
                "Predicted species: " + answer.species +
                "\\nConfidence: " + (answer.confidence * 100).toFixed(1) + "%";
            } catch (error) {
              result.textContent = "Error: " + error.message;
            }
          }
        );
      </script>
    </body>
    </html>
    """


@app.post("/predict")
def predict(data: FlowerInput):
    if model is None:
        raise RuntimeError("Model is not loaded.")

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica",
    }

    return {
        "species": species[prediction],
        "confidence": round(float(np.max(probabilities)), 4),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
