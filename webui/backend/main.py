import os
import sys
import tempfile
import zipfile
import json
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.chdir(project_root)

from src.state.poster_state import create_state
from src.workflow.pipeline import create_workflow_graph

app = FastAPI(title="PosterGen WebUI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict[str, Any]] = {}

job_logs: Dict[str, list] = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None

class PosterRequest(BaseModel):
    text_model: str
    vision_model: str
    poster_width: float
    poster_height: float

def get_available_models():
    return [
        "claude-sonnet-4-20250514",
        "gpt-4o-2024-08-06",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5v",
        "glm-4",
        "glm-4v"
    ]

def create_job_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"postgen_{timestamp}"
    job_dir = Path(tempfile.mkdtemp(prefix=f"{dir_name}_"))
    return job_dir

def add_job_log(job_id: str, message: str):
    if job_id not in job_logs:
        job_logs[job_id] = []
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    job_logs[job_id].append(log_entry)
    
    if len(job_logs[job_id]) > 50:
        job_logs[job_id] = job_logs[job_id][-50:]

async def run_poster_generation(job_id: str, config: dict, files: dict):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Initializing pipeline..."
        add_job_log(job_id, "Starting poster generation pipeline")
        
        job_dir = create_job_directory()
        add_job_log(job_id, f"Created working directory: {job_dir.name}")
        
        pdf_path = job_dir / "paper.pdf"
        logo_path = job_dir / "logo.png"
        aff_logo_path = job_dir / "aff_logo.png"
        
        with open(pdf_path, "wb") as f:
            f.write(files["pdf"])
        with open(logo_path, "wb") as f:
            f.write(files["logo"])
        with open(aff_logo_path, "wb") as f:
            f.write(files["aff_logo"])
        
        add_job_log(job_id, "Uploaded files saved successfully")
            
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Setting up workflow..."
        add_job_log(job_id, "Configuring pipeline workflow")
        
        state = create_state(
            pdf_path=str(pdf_path),
            text_model=config["text_model"],
            vision_model=config["vision_model"],
            width=int(config["poster_width"]),
            height=int(config["poster_height"]),
            url="",
            logo_path=str(logo_path),
            aff_logo_path=str(aff_logo_path)
        )
        
        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Running poster generation..."
        add_job_log(job_id, "Initializing multi-agent workflow")
        
        graph = create_workflow_graph()
        workflow = graph.compile()
        add_job_log(job_id, "Workflow graph compiled successfully")
        
        jobs[job_id]["progress"] = 40
        jobs[job_id]["message"] = "Analyzing paper content..."
        add_job_log(job_id, "Starting document analysis and content extraction")
        
        final_state = workflow.invoke(state)
        add_job_log(job_id, "Workflow execution completed")
        
        jobs[job_id]["progress"] = 80
        jobs[job_id]["message"] = "Generating final outputs..."
        add_job_log(job_id, "Creating final poster files")
        
        if final_state.get("errors"):
            raise Exception(f"Pipeline errors: {final_state['errors']}")
            
        output_dir = Path(final_state["output_dir"])
        zip_path = job_dir / f"{final_state['poster_name']}_output.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if output_dir.exists():
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)
        
        add_job_log(job_id, f"ZIP package created: {zip_path.name}")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Poster generation completed!"
        jobs[job_id]["output_file"] = str(zip_path)
        jobs[job_id]["poster_name"] = final_state["poster_name"]
        jobs[job_id]["output_dir"] = str(output_dir)
        add_job_log(job_id, "✅ Poster generation completed successfully!")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Error: {str(e)}"
        add_job_log(job_id, f"❌ Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "PosterGen WebUI API"}

@app.get("/models")
async def get_models():
    """Get available models for dropdowns"""
    models = get_available_models()
    return {"models": models}

@app.post("/generate", response_model=JobStatus)
async def generate_poster(
    background_tasks: BackgroundTasks,
    text_model: str = Form(...),
    vision_model: str = Form(...),
    poster_width: float = Form(...),
    poster_height: float = Form(...),
    pdf_file: UploadFile = File(...),
    logo_file: UploadFile = File(...),
    aff_logo_file: UploadFile = File(...)
):
    
    available_models = get_available_models()
    if text_model not in available_models:
        raise HTTPException(status_code=400, detail=f"Invalid text model: {text_model}")
    if vision_model not in available_models:
        raise HTTPException(status_code=400, detail=f"Invalid vision model: {vision_model}")
    
    ratio = poster_width / poster_height
    if ratio > 2 or ratio < 1.4:
        raise HTTPException(status_code=400, detail=f"Poster ratio {ratio:.2f} out of range (1.4-2.0)")
    
    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Paper must be a PDF file")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Job created, starting soon..."
    }
    
    job_logs[job_id] = []
    add_job_log(job_id, "Job created and queued for processing")
    
    pdf_content = await pdf_file.read()
    logo_content = await logo_file.read()
    aff_logo_content = await aff_logo_file.read()
    
    config = {
        "text_model": text_model,
        "vision_model": vision_model,
        "poster_width": poster_width,
        "poster_height": poster_height
    }
    
    files = {
        "pdf": pdf_content,
        "logo": logo_content,
        "aff_logo": aff_logo_content
    }
    
    background_tasks.add_task(run_poster_generation, job_id, config, files)
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        message="Job started..."
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        error=job.get("error")
    )

@app.get("/logs/{job_id}")
async def get_job_logs(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logs = job_logs.get(job_id, [])
    return {"logs": logs}

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_file = job.get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    poster_name = job.get("poster_name", "poster")
    filename = f"{poster_name}_poster_output.zip"
    
    return FileResponse(
        path=output_file,
        media_type="application/zip",
        filename=filename
    )

@app.get("/poster/{job_id}")
async def get_poster_image(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    poster_name = job.get("poster_name", "poster")
    png_path = Path("output") / poster_name / f"{poster_name}.png"
    
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="Poster image not found")
    
    return FileResponse(
        path=str(png_path),
        media_type="image/png"
    )

@app.get("/files/{job_id}")
async def get_json_files(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Use the specific job's output directory
    output_dir = job.get("output_dir")
    if not output_dir:
        return {"files": {}}
    
    output_path = Path(output_dir)
    content_path = output_path / "content"
    
    json_files = {}
    
    # Only search in the specific job's content directory
    if content_path.exists():
        for json_file in content_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_files[json_file.name] = json.load(f)
            except Exception:
                continue
    
    return {"files": json_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)