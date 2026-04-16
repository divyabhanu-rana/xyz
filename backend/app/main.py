import os
import traceback
import threading
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from dotenv import load_dotenv
import json
import time
from contextlib import asynccontextmanager

# --- Load environment variables from .env file in parent directory (backend/.env)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from .rag_pipeline import generate_material, CHAPTER_FILE_MAP
from .export import export_text
from ollama_client import query_deepseek

# -- Import auto_vectorizer and file_watcher --
from .auto_vectorizer import auto_vectorize
from .file_watcher import start_watcher

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

_observer = None
_watcher_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _observer, _watcher_thread

    # Startup: batch vectorize + start watcher thread
    auto_vectorize()

    def run_watcher():
        global _observer
        _observer = start_watcher()
        while _observer and _observer.is_alive():
            time.sleep(1)

    _watcher_thread = threading.Thread(target=run_watcher, daemon=True)
    _watcher_thread.start()

    yield  # App runs here

    # Shutdown: stop watcher if running
    if _observer:
        _observer.stop()
        _observer.join(timeout=5)
        _observer = None

app = FastAPI(
    title="AI Material Generator",
    description="Generate educational materials (Question Papers/Worksheets/Lesson Plans) for Grades 1-12 using a RAG (Retrieval-Augmented Generation) pipeline by utilising Deepseek.",
    version="1.1.0",
    lifespan=lifespan
)

# --- CORS Middleware for local frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## --- DATA MODELS ---
class GenerateRequest(BaseModel):
    grade: str  # "Grade 1", ..., "Grade 12"
    chapter: Union[str, List[str]]  # Supports both single string and list of chapters
    material_type: str  # "Question Paper" or "Worksheet" or "Lesson Plan"
    difficulty: str  # "Easy", "Medium", "Difficult"
    stream: Optional[str] = None  # Only for Grades 11 and 12
    max_marks: Optional[int] = None  # Only required for Question Paper

class GenerateResponse(BaseModel):
    output: str

class ExportRequest(BaseModel):
    text: str
    filetype: str = "pdf"  # "pdf" or "docx"

class ExportResponse(BaseModel):
    file_path: str

class DeepseekRequest(BaseModel):
    materialType: Optional[str] = "worksheet"
    grade: Optional[str] = "X"
    chapter: Optional[str] = "General"
    difficulty: Optional[str] = "medium"

class DeepseekResponse(BaseModel):
    output: str

## --- ENDPOINTS ---

@app.get("/api/grades", response_model=List[str])
def get_grades():
    return [f"Grade {i}" for i in range(1, 13)]

@app.get("/api/material_types", response_model=List[str])
def get_material_types():
    return ["Question Paper", "Worksheet", "Lesson Plan"]

@app.get("/api/difficulty_levels", response_model=List[str])
def get_difficulty_levels():
    return ["Easy", "Medium", "Difficult"]

@app.get("/api/chapters", response_model=List[str])
def get_chapters(grade: str = Query(..., description="Grade like '2' or 'Grade 2'")):
    grade_num = ''.join(filter(str.isdigit, grade))
    if not grade_num:
        raise HTTPException(status_code=400, detail="Invalid grade format.")

    chapters = sorted({
        ch for (g, ch) in CHAPTER_FILE_MAP.keys()
        if g == grade_num and ch.strip()
    })

    return chapters

def build_prompt(data: DeepseekRequest) -> str:
    return (
        f"Generate a {data.materialType or 'worksheet'} for grade {data.grade or 'X'},"
        f' chapter "{data.chapter or "General"}", with {data.difficulty or "medium"} difficulty.'
    )

@app.post("/api/generate", response_model=GenerateResponse)
def generate(generate_req: GenerateRequest):
    # Validate max_marks for Question Paper
    if generate_req.material_type.strip().lower() == "question paper" and not generate_req.max_marks:
        raise HTTPException(status_code=400, detail="max_marks is required for Question Paper.")

    # --- Parse chapters as a list ---
    chapters = generate_req.chapter
    if isinstance(chapters, str):
        # Split on commas, strip whitespace, remove empty entries
        chapters_list = [c.strip() for c in chapters.split(",") if c.strip()]
    elif isinstance(chapters, list):
        chapters_list = [c.strip() for c in chapters if isinstance(c, str) and c.strip()]
    else:
        chapters_list = []

    # Make a copy of the request with chapters as a list
    updated_generate_req = generate_req.copy(update={"chapter": chapters_list})

    try:
        output = generate_material(updated_generate_req)
        return {"output": output}
    except Exception as excep:
        print("Error in /api/generate:", excep)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(excep))

@app.post("/api/deepseek_generate", response_model=DeepseekResponse)
def deepseek_generate(deepseek_req: DeepseekRequest):
    """Endpoint migrated from Flask for Deepseek prompt-based generation."""
    try:
        prompt = build_prompt(deepseek_req)
        result = query_deepseek(prompt)
        return {"output": result}
    except Exception as excep:
        print("Error in /api/deepseek_generate:", excep)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(excep))

@app.post("/api/export", response_model=ExportResponse)
def export(export_req: ExportRequest):
    try:
        file_path = export_text(export_req.text, export_req.filetype)
        return {"file_path": file_path}
    except Exception as excep:
        print("Error in /api/export:", excep)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(excep))

@app.get("/api/download")
def download_file(file_path: str):
    try:
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
    except Exception as excep:
        raise HTTPException(status_code=404, detail=f"File not found: {excep}")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# ----------- STREAMING PROGRESS ENDPOINT -----------

@app.get("/api/generate_stream")
async def generate_stream(
    grade: str = Query(..., description="Grade number, e.g. '10'"),
    chapter: str = Query(..., description="Comma-separated list of chapters"),
    material_type: str = Query(..., description="Material type (Question Paper, Worksheet, Lesson Plan)"),
    difficulty: str = Query(..., description="Difficulty (Easy, Medium, Difficult)"),
    stream: Optional[str] = Query(None, description="Stream for 11/12"),
    max_marks: Optional[int] = Query(None, description="Maximum marks for Question Paper")
):
    """
    Streams progress updates and the final output for the progress bar.
    On the frontend, use EventSource to listen to /api/generate_stream and update the progress bar accordingly.
    """
    async def event_generator():
        # Simulate progress
        total_steps = 8
        for i in range(total_steps):
            progress = int((i / (total_steps - 1)) * 90)  # up to 90%
            yield f"data: {json.dumps({'progress': progress})}\n\n"
            await asyncio.sleep(0.8)

        # Final generation step
        from types import SimpleNamespace
        req = SimpleNamespace(
            grade=grade,
            chapter=[c.strip() for c in chapter.split(",") if c.strip()],
            material_type=material_type,
            difficulty=difficulty,
            stream=stream,
            max_marks=max_marks
        )
        try:
            output = generate_material(req)
            yield f"data: {json.dumps({'progress': 100, 'output': output})}\n\n"
        except Exception as ex:
            yield f"data: {json.dumps({'error': str(ex)})}\n\n"

    import asyncio
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)