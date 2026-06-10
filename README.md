# HealthPredictAI 🏥

An AI-powered clinical dashboard and Retrieval-Augmented Generation (RAG) assistant designed to predict 30-day hospital readmission risks. The application integrates machine learning pipelines with natural language interfaces to give clinicians predictive risk alerts and conversational access to patient discharge summaries.

## 🌟 Key Features

1. **Predictive Risk Assessment**:
   * Evaluates patient records using a trained **MLP (Multi-Layer Perceptron) Neural Network** pipeline achieving an average cross-validated **ROC-AUC of 0.9733**.
2. **Explainable AI (XAI)**:
   * Translates complex model parameters into patient-specific risk factor checklists (e.g., elevated creatinine, comorbidity counts, extended stays) sorted by impact.
3. **RAG Clinical Assistant**:
   * Features a sidebar chat assistant that queries patient discharge summaries using vector embeddings, with an offline rule-based fallback if no LLM API key is configured.
4. **Premium Dark Dashboard**:
   * A modern, single-page React interface styled with custom CSS variables, glassmorphic layouts, animated probability gauges, registry filters, and responsive grids.

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, Uvicorn, Scikit-Learn, XGBoost, Pandas, NumPy, Pydantic, Python-dotenv
* **Frontend**: React (Vite), Vanilla CSS (custom properties & HSL color palette), custom SVG iconography

---

## 📂 Project Structure

```text
healthpredictai/
├── backend/
│   ├── data/                 # Generated patient CSV and clinical discharge summaries
│   ├── models/               # Pickled MLP pipeline and feature importance data
│   ├── generate_data.py      # Script to generate synthetic medical cohort
│   ├── train_model.py        # ML training and evaluation script (5-fold CV)
│   ├── rag_pipeline.py       # Embedding vector database compilation and query engine
│   ├── main.py               # FastAPI backend server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Dashboard registry, metrics, and chat logic
│   │   ├── App.css           # Styling system, variables, layouts, and animations
│   │   ├── main.jsx          # React entrypoint
│   │   └── index.css         # Baseline css reset
│   ├── index.html            # Main HTML wrapper (SEO meta and Inter fonts)
│   ├── vite.config.js        # Vite config with dev proxy to port 8000
│   └── package.json          # Node dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js v18+ & npm

### 1. Backend Setup & Run

Navigate to the `backend` directory:
```bash
cd backend
```

Activate the virtual environment (Windows):
```bash
.\venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Generate patient records & discharge summaries:
```bash
python generate_data.py
```

Train the models & output feature importances:
```bash
python train_model.py
```

Initialize the vector database store:
```bash
python rag_pipeline.py
```

Start the FastAPI backend server:
```bash
python main.py
```
*The API is now running locally at `http://127.0.0.1:8000`.*

### 2. Frontend Setup & Run

Open a new terminal window, navigate to the `frontend` directory:
```bash
cd frontend
```

Install Node packages:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```
*The dashboard will be live at `http://localhost:5173/`.*

---

## 📡 API Reference

* **`GET /api/patients`**: Retrieves the list of patients in the registry.
* **`GET /api/patients/{patient_id}/risk`**: Runs predictive model inference on the selected patient and returns their readmission probability and top risk factors.
* **`POST /api/patients/{patient_id}/chat`**: Sends a question about the patient's record to the RAG chart assistant.

---

## ⚠️ Disclaimer
This application is a clinical simulation project utilizing synthetic patient data. It is intended for demonstrative/portfolio purposes and is not certified for real-world diagnostic or medical support workflows.
