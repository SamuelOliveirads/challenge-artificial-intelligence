from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.llm.create_rag_db import update_chroma_db
from src.llm.llm_model import StudyJourney

description = """
StudyJourney API

## Overview

Provides interactive learning assistance via the StudyJourney assistant,
tailoring content recommendations based on user's knowledge gaps.

## Endpoints

### `POST /query`

Accepts questions, returns tailored responses and relevant document excerpts.

#### Request

- Method: `POST`
- URL: `/query`
- Body (JSON):
  - `question` (str): The user's question.

#### Response

- Status: `200 OK`
- Body (JSON):
  - `message` (str): The response message from the assistant.

#### Example

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json"
-d '{"question": "Como posso aprender fundamentos de programação?",
"stage": "main"}'
"""

app = FastAPI(title="StudyJourney API", description=description, version="1.0.0")

retriever = update_chroma_db()
study_journey = StudyJourney(retriever=retriever)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    message: str


@app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    try:
        message = study_journey.get_answer(question=request.question)
        return QueryResponse(message=message["text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
