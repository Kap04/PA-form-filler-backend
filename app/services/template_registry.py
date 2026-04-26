from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models import TemplateProfile
from app.services.template_detection import TemplateDetector


class TemplateRegistry:
    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.settings = get_settings()
        self.registry_path = Path(registry_path or self.settings.template_registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")
        self.detector = TemplateDetector()

    def list_templates(self) -> list[TemplateProfile]:
        raw_entries = self._read_all()
        return [TemplateProfile.model_validate(entry) for entry in raw_entries]

    def register_pdf(self, template_pdf_path: str | Path, template_name: str | None = None) -> TemplateProfile:
        profile = self.detector.build_profile(template_pdf_path, template_name=template_name)
        entries = self._read_all()
        entries = [entry for entry in entries if entry.get("signature") != profile.signature]
        entries.append(profile.model_dump())
        self.registry_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return profile

    def detect_match(self, template_pdf_path: str | Path) -> tuple[TemplateProfile, float] | None:
        if not self.registry_path.exists():
            return None
        current_profile = self.detector.build_profile(template_pdf_path)
        known_templates = self.list_templates()
        detected = self.detector.detect_best_template(current_profile, known_templates)
        if detected is None:
            return None
        return detected.profile, detected.match_score

    def detect_or_register(self, template_pdf_path: str | Path, template_name: str | None = None) -> tuple[TemplateProfile, float]:
        current_profile = self.detector.build_profile(template_pdf_path, template_name=template_name)
        known_templates = self.list_templates()
        detected = self.detector.detect_best_template(current_profile, known_templates)
        if detected and detected.match_score >= 0.6:
            return detected.profile, detected.match_score
        registered = self.register_pdf(template_pdf_path, template_name=template_name)
        return registered, 1.0

    def _read_all(self) -> list[dict[str, Any]]:
        content = self.registry_path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)
