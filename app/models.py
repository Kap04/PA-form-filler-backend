from typing import Any
from pydantic import BaseModel, Field


class ExtractedField(BaseModel):
    name: str
    value: str = ""
    confidence: float = 0.0
    source: str = ""


class TemplateField(BaseModel):
    name: str
    label: str = ""
    field_type: str = "text"
    required: bool = False
    max_length: int | None = None
    options: list[str] = Field(default_factory=list)


class TemplateProfile(BaseModel):
    template_name: str
    field_count: int
    page_count: int
    fields: list[TemplateField]
    signature: str


class PipelineResult(BaseModel):
    job_id: str
    template_name: str
    extracted_fields: list[ExtractedField]
    output_pdf_path: str
    tracker_entry: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


class ProcessResponse(BaseModel):
    job_id: str
    template_name: str
    extracted_fields: list[ExtractedField]
    download_path: str
    editable: bool = True
    tracker_entry: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    job_id: str
    template_name: str
    extracted_fields: list[ExtractedField]
    draft_download_path: str
    tracker_entry: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


class FillRequest(BaseModel):
    field_values: dict[str, str] = Field(default_factory=dict)
