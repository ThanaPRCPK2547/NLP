# Data Engineering RAG Agent PoC

โปรเจกต์นี้เป็น Proof of Concept สำหรับ AI Agent ที่กำหนด role ของ model เป็น **Senior Data Engineer** และจำกัดข้อมูลให้อยู่เฉพาะหมวดหมู่ของ **Data Engineering** เท่านั้น

## วิธีรัน

```bash
python3 -m pip install -r requirements.txt
export GOOGLE_API_KEY="your-google-api-key"
python3 20260425_ai_agent_v2.py --gui
```

ถ้ายังไม่มี API key ระบบจะใช้ deterministic fallback แทน LLM จริง ถ้ายังไม่ได้ติดตั้ง Gradio คำสั่ง `--gui` จะเปิด built-in web GUI ด้วย standard library อัตโนมัติ

ทดสอบ logic หลักผ่าน CLI:

```bash
python3 20260425_ai_agent_v2.py --demo
```

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

ถ้าคำถามหลุดจาก data engineering prompt จะบังคับให้ model ตอบแบบ reframe กลับมาเป็นปัญหา data engineering หรือถาม clarification แทน

## Rubric Mapping

| เกณฑ์ | สิ่งที่ทำในโปรเจกต์ |
|---|---|
| Module | แยก class/ฟังก์ชันเป็น `DynamicMemory`, `QueryEnhancer`, `SemanticRetriever`, `ToolMiddleware`, `DataEngineeringRAGAgent`, `OutputController` |
| Tool | Agent ใช้ tools หลายตัว เช่น scope classification, architecture recommendation, capacity estimate, pipeline plan, quality controls, governance controls |
| Retrieval | ใช้ semantic search ผ่าน Gemini embedding เมื่อมี API key และ fallback เป็น local TF-IDF โดยค้นเฉพาะ knowledge base หมวด data engineering |
| Augmented | prompt รวม role senior data engineer, user profile, dynamic memory, retrieved context, selected categories และ tool trace |
| Generation | ใช้ Gemini LLM generate JSON answer และมี deterministic fallback สำหรับ demo |
| Wow Module | มี fallback, category validation, source validation, output schema control, exception-safe tool middleware |
| Wow Tool | มีมากกว่า 1 tool และ trace การเรียกใช้ tool ใน debug panel |
| Wow Retrieval | มี query expansion ภาษาไทย/อังกฤษ เช่น ดึงข้อมูล, แปลงข้อมูล, คุณภาพ, orchestration, streaming, governance |
| Wow Augmented | prompt ถูกออกแบบให้เหมาะกับบริบท data engineering และบังคับ allowed categories |
| Wow Generation | มี Gradio GUI, built-in web GUI fallback, structured output control, source/category validation |
| Wow Advanced | มี dynamic memory, middleware, prompt control, category-scoped retrieval, output schema control, input validation และ prompt-injection guard |

## Injection Guard

ระบบมีการป้องกันข้อมูล injection ในระดับ PoC:

- จำกัดขนาด request body และความยาวคำถาม
- sanitize control characters จาก user input
- validate ค่า enum ฝั่ง backend เช่น role level, project size, stack, answer style
- clamp `records_per_day` ให้อยู่ในช่วงที่กำหนด
- detect pattern เช่น `ignore previous instructions`, `reveal system prompt`, `<script>`, `javascript:`, `onerror=`
- prompt ระบุชัดว่าคำถามและ memory เป็น untrusted data
- output ใช้ `textContent` ใน built-in website ไม่ใช้ `innerHTML`
- API response มี security headers เช่น `X-Content-Type-Options`, `Referrer-Policy`, `Cache-Control`, `Content-Security-Policy`

## หมายเหตุความปลอดภัย

โค้ดไม่ฝัง API key ในไฟล์ ให้ใช้ environment variable หรือ `.env`:

```env
GOOGLE_API_KEY=your-google-api-key
```
