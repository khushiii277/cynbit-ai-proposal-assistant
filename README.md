# Cynbit AI Proposal & SOW Assistant

A practical RAG proof of concept that retrieves relevant reference projects and drafts a technical proposal / Statement of Work (SOW) from a client RFP.

> **Data honesty:** `data/past_projects.json` contains synthetic demonstration records, not verified Cynbit client projects. Do not describe them as actual Cynbit engagements. The tool is a draft assistant and requires human review.

## Features

- Upload PDF or TXT RFPs, or use `data/sample_rfp.txt`
- Semantic retrieval with Sentence Transformers + FAISS
- Keyword fallback if embedding dependencies/model downloads are unavailable
- Optional OpenAI-compatible LLM integration
- Transparent template mode if no API key is supplied
- Shows retrieved evidence and reference IDs
- Editable Markdown output and download
- Explicit assumptions, risks, questions, and human-review caveats

## Project structure

```text
cynbit-ai-proposal-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   ├── past_projects.json
│   └── sample_rfp.txt
└── src/
    ├── __init__.py
    ├── retriever.py
    └── generator.py
```

## Requirements

- Python 3.10 or newer
- Internet access on first semantic-search run to download the Sentence Transformers model
- Optional LLM API key for model-generated drafts

## Run locally

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `faiss-cpu` is not available for your Python/platform, use a supported Python version (typically Python 3.10–3.12) or remove FAISS and use keyword fallback after adapting `retriever.py`.

### 3. Configure the LLM (optional)

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
# Optional OpenAI-compatible provider:
# OPENAI_BASE_URL=https://your-provider.example/v1
```

Never commit `.env` or expose an API key in screenshots or public repositories.

If no key is configured, the app runs in template demo mode. It does not pretend that a model was called.

### 4. Launch Streamlit

From the repository root:

```bash
streamlit run app.py
```

Open the local URL printed in your terminal (normally `http://localhost:8501`).

## How it works

1. Read the RFP text.
2. Encode reference projects with `sentence-transformers/all-MiniLM-L6-v2`.
3. Use FAISS inner-product search over normalized vectors (cosine similarity).
4. Pass the RFP and retrieved records to the LLM as context.
5. Ask the LLM to draft a proposal while distinguishing requirements, assumptions, and unknowns.
6. Display the retrieved references for human verification.

## Example workflow

1. Open the app.
2. Click **Load sample RFP**.
3. Click **Retrieve references and draft proposal**.
4. Inspect evidence in the right-hand panel.
5. Download the Markdown proposal and review it before reuse.

## Security and responsible use

- Only upload documents you are authorized to process.
- Do not upload confidential client documents to an external model without organizational approval.
- This prototype does not include authentication or production-grade access controls.
- Treat retrieved documents and RFP content as untrusted input.
- Estimates are illustrative until calibrated against verified historical delivery data.
- A human must approve the final proposal and any commercial estimate.

## Known limitations

- PDF extraction supports selectable text, not scanned OCR.
- Mock projects are synthetic.
- No automatic email, CRM integration, contract generation, or client submission.
- Retrieval quality and generation quality must be evaluated before production use.

## Suggested next improvements

- Add a reviewed evaluation set and metrics for retrieval relevance, requirement coverage, citation support, and latency.
- Use structured output schemas for requirements, work packages, milestones, risks, and estimates.
- Add DOCX export, authentication, role-based access, audit logging, and document-retention controls.
- Calibrate estimates using verified project effort data rather than synthetic examples.
