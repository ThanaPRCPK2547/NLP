# DataArchitect AI

เว็บสำหรับใช้งาน Data Engineering RAG Agent ผ่าน Vercel

## โครงสร้างสำหรับ Deploy

- `public/index.html` เว็บหน้าเดียวที่เรียก API จริงผ่าน `/api/ask`
- `api/ask.py` Python Serverless Function สำหรับเรียก `DataEngineeringRAGAgent`
- `model/20260425_ai_agent_v2.py` logic หลักของ RAG agent
- `vercel.json` config สำหรับ Vercel
- `requirements.txt` dependency ฝั่ง serverless

## รันแบบ Local

ถ้าต้องการทดสอบ logic หลัก:

```bash
python3 model/20260425_ai_agent_v2.py --demo
```

ถ้าต้องการใช้เว็บแบบเดียวกับ Vercel:

```bash
vercel dev
```

จากนั้นเปิด URL ที่ Vercel CLI แสดง แล้วถามผ่านหน้าเว็บได้เลย

## Deploy ไป Vercel

1. ติดตั้ง Vercel CLI ถ้ายังไม่มี:

```bash
npm i -g vercel
```

2. Login และ deploy จาก root ของโปรเจกต์:

```bash
vercel login
vercel
```

3. ตั้งค่า environment variable ใน Vercel Project Settings:

```env
GOOGLE_API_KEY=your-google-api-key
```

ถ้าไม่ได้ตั้ง `GOOGLE_API_KEY` ระบบยังใช้งานได้ด้วย deterministic fallback แต่จะไม่เรียก Gemini จริง

4. Deploy production:

```bash
vercel --prod
```

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

Response จะมี `markdown`, `structured`, และ `debug` สำหรับแสดงคำตอบและ trace ของ agent
