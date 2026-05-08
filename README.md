# DataArchitect AI

เว็บสำหรับใช้งาน Data Engineering LangGraph ReAct Agent ผ่าน Vercel

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph v0.2–1.x (ReAct) |
| LLM | Gemini 2.5 Flash (via `langchain-google-genai`) |
| Vector Store | Qdrant (local disk persistence) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Tracing | Langfuse v2.x (optional) |

## Project Structure

- `public/index.html` Single-page web UI calling `/api/ask`
- `api/ask.py` Async Python Serverless Function wrapping `DataEngineeringAgent`
- `model/model.py` Core LangGraph ReAct agent with Qdrant + Langfuse
- `vercel.json` Vercel config (60s timeout, security headers)
- `requirements.txt` Serverless dependencies
- `model/requirements.txt` Full dependencies (including Gradio for local GUI)

## Local Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r model/requirements.txt
```

> **Note:** Pin versions are enforced in `requirements.txt`. If you already installed newer packages that break compatibility, re-run the command above to downgrade to compatible versions.

### 3. Set environment variables (`.env`)

```env
GOOGLE_API_KEY=your-google-api-key
LANGFUSE_PUBLIC_KEY=pk-lf-...       # optional
LANGFUSE_SECRET_KEY=sk-lf-...       # optional
LANGFUSE_HOST=https://cloud.langfuse.com
```

Only `GOOGLE_API_KEY` is required. Langfuse keys are optional — tracing is skipped when absent.

### 4. Run CLI demo

```bash
python3 model/model.py --demo
```

Qdrant is auto-initialized on first run — knowledge base (11 DE categories) is seeded to `./qdrant_db/`.

### 5. Run Vercel dev server

```bash
vercel dev
```

## Deploy to Vercel

1. Install Vercel CLI:

```bash
npm i -g vercel
```

2. Login and deploy:

```bash
vercel login
vercel
```

3. Set environment variables in Vercel Project Settings:

```env
GOOGLE_API_KEY=your-google-api-key
```

4. Deploy production:

```bash
vercel --prod
```

## Langfuse Tracing

Requires `langfuse>=2.55.0,<3.0.0` (pinned to v2.x API). If you have a newer version installed, re-run `pip install -r model/requirements.txt` to downgrade.

ทุกครั้งที่เรียก agent trace จะถูกส่งไปยัง Langfuse Cloud ประกอบด้วย:
- LLM calls (prompt, response, token usage)
- Tool invocations (input, output)
- Agent thought process
- User question (trace input) และ structured answer (trace output)

URL ของ trace จะแสดงใน `debug.trace_url` ของ API response

## API

`POST /api/ask`

```json
{
  "question": "ออกแบบ pipeline รับข้อมูล API รายวัน 1 ล้าน record พร้อม data quality",
  "name": "Thanakorn",
  "role_level": "mid",
  "project_size": "medium",
  "records_per_day": 1000000,
  "preferred_stack": "cloud-agnostic",
  "answer_style": "practical"
}
```

Response: `{ "markdown": "...", "structured": {...}, "debug": {...} }`
