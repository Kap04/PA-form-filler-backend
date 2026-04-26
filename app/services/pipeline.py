from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.config import get_settings
from app.models import ExtractedField, PipelineResult
from app.services.mistral_client import MistralClient
from app.services.pdf_extractor import PdfExtractor
from app.services.pdf_form_filler import PdfFormFiller
from app.services.pdf_overlay import PdfOverlay
from app.services.template_detection import TemplateDetector
from app.services.template_registry import TemplateRegistry
from app.services.tracker import TrackerStore


class PAPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.extractor = PdfExtractor()
        self.template_detector = TemplateDetector()
        self.form_filler = PdfFormFiller()
        self.overlay_filler = PdfOverlay()
        self.template_registry = TemplateRegistry()
        self.tracker = TrackerStore(self.settings.tracker_path)
        self.mistral = MistralClient()

    def close(self) -> None:
        self.mistral.close()

    def analyze(self, emr_pdf_path: str | Path, form_pdf_path: str | Path, job_id: str | None = None) -> dict[str, object]:
        job_id = job_id or str(uuid4())
        emr_text = self.extractor.extract_text(emr_pdf_path)
        form_schema = self.form_filler.extract_form_schema(form_pdf_path)
        profile, match_score = self.template_registry.detect_or_register(form_pdf_path, template_name=form_schema["template_name"])

        # Stage 1: extract target labels from the PA form (now with AI-determined types)
        form_labels = self.overlay_filler.extract_form_labels(form_pdf_path)
        label_metadata = self.overlay_filler.label_metadata
        print(f"DEBUG: analyze() form_labels = {form_labels}")
        print(f"DEBUG: analyze() label_metadata = {label_metadata}")

        # Stage 2: extract source facts from the EMR
        emr_facts = self.mistral.extract_emr_facts(emr_text)
        print(f"DEBUG: analyze() emr_facts = {emr_facts}")

        # Stage 3: map source facts to target labels (now with label type info)
        field_values = self.mistral.map_emr_facts_to_form_labels(emr_facts, form_labels, label_metadata)
        print(f"DEBUG: analyze() mapped_field_values = {field_values}")

        # Create extracted fields list for UI
        extracted_fields = [
            ExtractedField(
                name=label,
                value=field_values.get(label, ""),
                confidence=0.8 if field_values.get(label) else 0.0,
                source="source-to-target mapping",
            )
            for label in form_labels
        ]
        notes = [
            f"Extracted {len([f for f in emr_facts.values() if f])} source facts from EMR and mapped to {len([f for f in field_values.values() if f])} form labels."
        ]
        if match_score < 1.0:
            notes.append(f"Template matched registry score {match_score:.2f}.")

        return {
            "job_id": job_id,
            "profile": profile,
            "form_schema": form_schema,
            "emr_facts": emr_facts,
            "extracted_fields": extracted_fields,
            "field_values": field_values,
            "notes": notes,
        }

    def fill(
        self,
        template_pdf_path: str | Path,
        field_values: dict[str, str],
        job_id: str | None = None,
        status: str = "completed",
    ) -> PipelineResult:
        job_id = job_id or str(uuid4())
        profile = self.template_detector.build_profile(template_pdf_path)
        output_name = f"{Path(template_pdf_path).stem}_{job_id}.pdf"
        output_pdf_path = self.settings.output_dir / output_name
        filled_pdf_path = self.overlay_filler.fill_form_with_overlay(template_pdf_path, field_values, output_pdf_path)
        tracker_entry = self.tracker.add_entry(
            job_id=job_id,
            status=status,
            template_name=profile.template_name,
            form_pdf=str(template_pdf_path),
            output_pdf=str(filled_pdf_path),
        )
        return PipelineResult(
            job_id=job_id,
            template_name=profile.template_name,
            extracted_fields=[],
            output_pdf_path=str(filled_pdf_path),
            tracker_entry=tracker_entry,
            notes=[],
        )

    def process(self, emr_pdf_path: str | Path, form_pdf_path: str | Path, job_id: str | None = None) -> PipelineResult:
        # Analyze performs: form label detection -> EMR fact extraction -> source/target mapping
        analysis = self.analyze(emr_pdf_path, form_pdf_path, job_id=job_id)
        field_values = analysis["field_values"]

        # Fill the form with the mapped values
        filled = self.fill(form_pdf_path, field_values, job_id=analysis["job_id"])
        return PipelineResult(
            job_id=filled.job_id,
            template_name=analysis["profile"].template_name,
            extracted_fields=analysis["extracted_fields"],
            output_pdf_path=filled.output_pdf_path,
            tracker_entry=filled.tracker_entry,
            notes=analysis["notes"],
        )
