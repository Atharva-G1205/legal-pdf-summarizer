<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Vite_8-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/MUI_7-007FFF?style=for-the-badge&logo=mui&logoColor=white" alt="MUI" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
</p>

<h1 align="center">⚖️ Legal PDF Summarizer</h1>

<p align="center">
  <strong>Upload a legal case document. Choose your summary depth. Get an AI-generated analysis — in seconds.</strong>
</p>

<p align="center">
  A full-stack application that combines <strong>extractive retrieval</strong> (InLegalBERT embeddings + cosine ranking) with <strong>abstractive generation</strong> (Legal-LED / Legal-Pegasus) to produce faithful, multi-level summaries of Indian legal judgments.
</p>

---

## ✨ Features

| Feature                 | Description                                        |
| ----------------------- | -------------------------------------------------- |
| 📄 **PDF Upload**       | Drag-and-drop or click to upload any legal PDF     |
| 🔍 **4 Summary Levels** | Executive · Detailed · Technical · Extractive-only |
| 🤖 **Hybrid Pipeline**  | Extractive retrieval → Abstractive generation      |
| 📥 **Download as PDF**  | Export your summary as a clean, formatted PDF      |
| ⚡ **GPU Accelerated**  | Runs on CUDA when available for fast inference     |

---

## 🏗️ Architecture

```
┌──────────────┐       POST /api/summarize        ┌──────────────────┐
│   React 19   │ ──────────────────────────────▶  │  FastAPI Backend │
│   + MUI 7    │ ◀──────────────────────────────  │                  │
│   (Vite 8)   │       JSON response              │  uvicorn :8000   │
└──────────────┘                                   └────────┬─────────┘
                                                            │
                                               ┌────────────▼────────────┐
                                               │   Summarization Pipeline │
                                               │                         │
                                               │  pdf_loader             │
                                               │  → preprocessor         │
                                               │  → chunker              │
                                               │  → embedder (InLegalBERT)│
                                               │  → retriever (cosine)   │
                                               │  → summarizer (LED)     │
                                               └─────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

- **Python 3.11+** — Core language
- **FastAPI** — Async REST API
- **PyTorch** — Model inference (CUDA / CPU)
- **HuggingFace Transformers** — LED & Pegasus models
- **pdfplumber + pytesseract** — PDF text extraction with OCR fallback
- **fpdf2** — Summary PDF generation

### Frontend

- **React 19** — UI framework
- **TypeScript** — Type safety
- **Vite 8** — Build tool & dev server
- **MUI 7 (Material UI)** — Component library
- **react-dropzone** — File upload

### ML Models

- **InLegalBERT** — Legal domain sentence embeddings
- **Legal-LED** (`allenai/led-base-16384`) — Long-document abstractive summarization
- **Legal-Pegasus** — Alternative abstractive model

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- **CUDA GPU** _(recommended)_ — works on CPU but slower

### 1 · Clone the repository

```bash
git clone https://github.com/Atharva-G1205/legal-pdf-summarizer.git
cd legal-pdf-summarizer
```

### 2 · Set up the backend

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3 · Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 4 · Run the application

Open **two terminals** from the project root:

**Terminal 1 — Backend API**

```bash
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend dev server**

```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## 📁 Project Structure

```
legal-pdf-summarizer/
├── backend/                  # FastAPI server
│   ├── main.py               #   App entry point
│   ├── api/
│   │   ├── summarize.py      #   /api/summarize endpoint
│   │   └── schemas.py        #   Pydantic models
│   └── services/
│       └── pdf_generator.py  #   Summary → PDF export
│
├── pipeline/                 # ML summarization pipeline
│   ├── pdf_loader.py         #   PDF text extraction
│   ├── preprocessor.py       #   Legal text preprocessing
│   ├── chunker.py            #   Sentence chunking
│   ├── embedder.py           #   InLegalBERT embeddings
│   ├── retriever.py          #   Cosine similarity ranking
│   ├── summarizer.py         #   Abstractive summary generation
│   ├── summarize.py          #   Pipeline orchestration
│   └── config.py             #   Summary level configs
│
├── frontend/                 # React + Vite SPA
│   ├── src/
│   │   ├── App.tsx            #   Main application
│   │   ├── components/        #   UI components
│   │   ├── services/api.ts    #   Backend API client
│   │   └── theme.ts           #   MUI theme
│   └── index.html
│
├── models/                   # Pre-trained model weights
│   ├── LED/IN_model/          #   Legal-LED checkpoint
│   └── Pegasus/               #   Legal-Pegasus checkpoint
│
├── data/                     # Datasets
│   └── dataset/IN-Abs/        #   Indian legal judgment-summary pairs
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📋 API Reference

### `POST /api/summarize`

Upload a PDF and get a summary.

| Parameter | Type   | Description                                                                       |
| --------- | ------ | --------------------------------------------------------------------------------- |
| `file`    | `File` | PDF document to summarize                                                         |
| `level`   | `int`  | Summary level: `1` (Executive), `2` (Detailed), `3` (Technical), `4` (Extractive) |

**Response:**

```json
{
  "summary": "The court held that ...",
  "level_name": "Executive Summary",
  "model_used": "led",
  "word_count": 142,
  "input_word_count": 5200
}
```

### `GET /health`

Health check endpoint — returns `{"status": "ok"}`.

---

## 📄 License

Built as part of Inhouse Internship under Intelligent NLP and Foundational Embeddings Research (INFER) Lab, PICT, Pune.

---
