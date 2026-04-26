from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz

from app.models import TemplateField, TemplateProfile


@dataclass(slots=True)
class DetectedTemplate:
    profile: TemplateProfile
    match_score: float


class TemplateDetector:
    def build_profile(self, form_pdf_path: str | Path, template_name: str | None = None) -> TemplateProfile:
        document = fitz.open(str(form_pdf_path))
        try:
            fields: list[TemplateField] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                for widget in page.widgets() or []:
                    fields.append(
                        TemplateField(
                            name=widget.field_name or f"page_{page_index + 1}_field_{len(fields) + 1}",
                            label=(widget.field_label or "").strip(),
                            field_type=str(widget.field_type or "text"),
                            required=bool(widget.field_flags & 2) if widget.field_flags is not None else False,
                            max_length=int(widget.max_len) if getattr(widget, "max_len", 0) else None,
                            options=list(widget.choice_values or []),
                        )
                    )

            template_name = template_name or self._derive_template_name(form_pdf_path)
            signature = self._signature_from_fields(fields, document.page_count)
            return TemplateProfile(
                template_name=template_name,
                field_count=len(fields),
                page_count=document.page_count,
                fields=fields,
                signature=signature,
            )
        finally:
            document.close()

    def detect_best_template(self, target_profile: TemplateProfile, candidates: list[TemplateProfile]) -> DetectedTemplate | None:
        if not candidates:
            return None
        scored = [(self._profile_similarity(target_profile, candidate), candidate) for candidate in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, top_candidate = scored[0]
        return DetectedTemplate(profile=top_candidate, match_score=top_score)

    def _signature_from_fields(self, fields: list[TemplateField], page_count: int) -> str:
        field_names = ",".join(sorted(field.name.lower() for field in fields))
        raw = f"{page_count}|{field_names}|{len(fields)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _profile_similarity(self, left: TemplateProfile, right: TemplateProfile) -> float:
        left_names = Counter(field.name.lower() for field in left.fields)
        right_names = Counter(field.name.lower() for field in right.fields)
        overlap = sum(min(left_names[name], right_names[name]) for name in set(left_names) | set(right_names))
        field_score = overlap / max(len(left.fields), len(right.fields), 1)
        page_score = 1.0 if left.page_count == right.page_count else 0.5
        return round(field_score * 0.8 + page_score * 0.2, 4)

    def _derive_template_name(self, form_pdf_path: str | Path) -> str:
        return Path(form_pdf_path).stem.replace("_", " ").title()
