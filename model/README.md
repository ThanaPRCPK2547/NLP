# Data Engineering RAG Agent — LangGraph ReAct + Qdrant + Langfuse

โปรเจกต์ AI Agent สำหรับ Data Engineering ใช้ LangGraph ReAct Agent กับ Qdrant vector store สำหรับ semantic search และ Langfuse สำหรับ tracing

## Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph v0.2–1.x (ReAct pattern) |
| LLM | Gemini 2.5 Flash (via `langchain-google-genai`) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | Qdrant (local disk persistence) |
| Tracing | Langfuse v2.x |

## วิธีรัน

```bash
pip install -r requirements.txt

# ตั้งค่า environment variables (ต้องมีอย่างน้อย GOOGLE_API_KEY)
export GOOGLE_API_KEY="your-google-api-key"
export LANGFUSE_PUBLIC_KEY="pk-..."       # ไม่ต้องตั้งก็ได้ (tracing จะถูกข้าม)
export LANGFUSE_SECRET_KEY="sk-..."       # ไม่ต้องตั้งก็ได้
export LANGFUSE_HOST="https://cloud.langfuse.com"

# CLI demo
python3 model.py --demo
```

Qdrant ทำงานแบบ local ไม่ต้องตั้งค่าเพิ่ม — ฐานความรู้ Data Engineering 11 หมวดจะถูก seed อัตโนมัติในครั้งแรกที่รัน

## Data Engineering Scope

ระบบ retrieval และ output control ถูกจำกัดให้อยู่ในหมวดต่อไปนี้:

- `ingestion`
- `storage_lakehouse`
- `transformation`
- `orchestration`
- `data_quality`
- `governance_security`
- `observability`
- `batch_processing`
- `stream_processing`
- `data_modeling`
- `performance_cost`

## LangGraph ReAct Agent Flow

```
User Message → Agent Node (Gemini + system prompt)
                  │
                  ├── calls tool → Tool Node → returns result → back to Agent Node
                  │
                  └── responds → Final structured JSON output
```

Agent tools:
1. `retrieve_knowledge` — semantic search on Qdrant
2. `web_search` — DuckDuckGo fallback
3. `classify_scope` — DE category classification
4. `recommend_architecture` — architecture recommendation
5. `estimate_capacity` — volume/capacity estimation
6. `build_pipeline_plan` — implementation steps
7. `quality_controls` — data quality checks
8. `governance_controls` — governance & security controls

## Langfuse Tracing

ทุกรอบการเรียก agent จะถูกบันทึก trace ไปยัง Langfuse รวมถึง:
- LLM calls (prompt + response)
- Tool invocations (input + output)
- Agent thought process
- Token usage

URL ของ trace จะแสดงใน `debug.trace_url` ของ response

## หมายเหตุ

- Qdrant ใช้ local disk persistence ที่ `./qdrant_db/`
