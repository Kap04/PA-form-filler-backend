from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models import ExtractResponse, FillRequest, ProcessResponse
from app.services.pipeline import PAPipeline
from app.services.tracker import TrackerStore

settings = get_settings()
app = FastAPI(title="PA SaaS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PAPipeline()
tracker = TrackerStore(settings.tracker_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/pdf-text")
def debug_pdf_text(file_path: str = "PA_form.pdf") -> dict[str, object]:
    """Extract all text from a PDF to understand form structure."""
    import fitz
    from pathlib import Path
    
    pdf_path = Path(settings.base_dir) / file_path
    if not pdf_path.exists():
        return {"error": f"File not found: {pdf_path}"}
    
    document = fitz.open(str(pdf_path))
    try:
        pages_text = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = page.get_text()
            pages_text.append({
                "page": page_index + 1,
                "text": text[:500],  # First 500 chars
            })
        return {"page_count": document.page_count, "pages": pages_text}
    finally:
        document.close()


@app.get("/debug/pdf-widgets")
def debug_pdf_widgets(file_path: str = "PA_form.pdf") -> dict[str, object]:
    """Debug endpoint to inspect PDF widget structure."""
    import fitz
    from pathlib import Path
    
    pdf_path = Path(settings.base_dir) / file_path
    if not pdf_path.exists():
        return {"error": f"File not found: {pdf_path}"}
    
    document = fitz.open(str(pdf_path))
    try:
        widgets_info = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            widgets = page.widgets() or []
            for i, widget in enumerate(widgets):
                widget_data = {
                    "index": i,
                    "field_name": widget.field_name,
                    "field_label": widget.field_label,
                    "field_type": widget.field_type,
                    "field_flags": widget.field_flags,
                    "rect": str(widget.rect) if hasattr(widget, 'rect') else None,
                    "all_attrs": [attr for attr in dir(widget) if not attr.startswith("_")][:20],
                }
                widgets_info.append(widget_data)
        return {"page_count": document.page_count, "widgets": widgets_info}
    finally:
        document.close()


@app.get("/debug/mistral")
def debug_mistral() -> dict[str, object]:
    return {
        "mistral_api_key_set": bool(settings.mistral_api_key),
        "mistral_api_key_length": len(settings.mistral_api_key) if settings.mistral_api_key else 0,
        "mistral_model": settings.mistral_model,
        "mistral_base_url": settings.mistral_base_url,
    }


@app.get("/tracker")
def get_tracker() -> list[dict[str, object]]:
    return tracker.list_entries()


@app.post("/extract", response_model=ExtractResponse)
async def extract_files(emr_pdf: UploadFile = File(...), pa_form_pdf: UploadFile = File(...)) -> ExtractResponse:
    job_id = str(uuid4())
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    emr_path = await _persist_upload(emr_pdf, job_dir / "emr.pdf")
    form_path = await _persist_upload(pa_form_pdf, job_dir / "form.pdf")

    analysis = pipeline.analyze(emr_path, form_path, job_id=job_id)
    draft_pdf_name = f"{Path(form_path).stem}_{job_id}_draft.pdf"
    draft_pdf_path = settings.output_dir / draft_pdf_name
    pipeline.form_filler.fill_pdf(form_path, analysis["field_values"], draft_pdf_path)

    tracker_entry = tracker.add_entry(
        job_id=job_id,
        status="review_ready",
        template_name=analysis["profile"].template_name,
        emr_pdf=str(emr_path),
        form_pdf=str(form_path),
        draft_pdf=str(draft_pdf_path),
    )
    return ExtractResponse(
        job_id=job_id,
        template_name=analysis["profile"].template_name,
        extracted_fields=analysis["extracted_fields"],
        draft_download_path=f"/download/{draft_pdf_path.name}",
        tracker_entry=tracker_entry,
        notes=analysis["notes"],
    )


@app.post("/fill/{job_id}", response_model=ProcessResponse)
async def fill_job(job_id: str, payload: FillRequest = Body(...)) -> ProcessResponse:
    job_dir = _job_dir(job_id)
    form_path = job_dir / "form.pdf"
    if not form_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    result = pipeline.fill(form_path, payload.field_values, job_id=job_id, status="ready_to_submit")
    return ProcessResponse(
        job_id=result.job_id,
        template_name=result.template_name,
        extracted_fields=[],
        download_path=f"/download/{Path(result.output_pdf_path).name}",
        editable=True,
        tracker_entry=result.tracker_entry,
        notes=["Fields were updated from the review screen and the editable PDF was regenerated."],
    )


@app.post("/process", response_model=ProcessResponse)
async def process_files(emr_pdf: UploadFile = File(...), pa_form_pdf: UploadFile = File(...)) -> ProcessResponse:
    job_id = str(uuid4())
    job_dir = _job_dir(job_id)
    emr_path = await _persist_upload(emr_pdf, job_dir / "emr.pdf")
    form_path = await _persist_upload(pa_form_pdf, job_dir / "form.pdf")
    result = pipeline.process(emr_path, form_path, job_id=job_id)
    return ProcessResponse(
        job_id=result.job_id,
        template_name=result.template_name,
        extracted_fields=result.extracted_fields,
        download_path=f"/download/{Path(result.output_pdf_path).name}",
        editable=True,
        tracker_entry=result.tracker_entry,
        notes=result.notes,
    )


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = settings.output_dir / filename
    return FileResponse(path=str(file_path), filename=filename, media_type="application/pdf")


async def _persist_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    await upload.close()
    return destination


def _job_dir(job_id: str) -> Path:
    return settings.jobs_dir / job_id
