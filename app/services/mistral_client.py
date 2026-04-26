from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings


@dataclass(slots=True)
class MistralExtractionResult:
    raw_response: str
    parsed: dict[str, Any]


class MistralClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.Client(timeout=120.0)

    def close(self) -> None:
        self._client.close()

    def _normalize_label(self, label: str) -> str:
        """Normalize a label by removing special characters and converting to lowercase."""
        return re.sub(r"[^a-z0-9]+", "", label.lower())

    def extract_form_labels_with_ai(self, form_text: str) -> list[dict[str, Any]]:
        """Use AI to extract form labels with their types and metadata.
        Returns list of: {"name": string, "type": "text|choice|date|code", "options": [string], "context": string}
        """
        if not self.settings.mistral_api_key:
            return []

        prompt = (
            "Analyze this Prior Authorization form and extract all form labels/questions. "
            "Return JSON as {\"labels\": [{\"name\": string, \"type\": string, \"options\": [string], \"context\": string}]}.\n\n"
            "For each label, provide:\n"
            "- name: The exact label/question text as it appears\n"
            "- type: 'text' (free text input), 'choice' (checkbox/radio), 'date' (date field), or 'code' (code like ICD-10)\n"
            "- options: List of choices if type='choice', empty list otherwise. Include exact option text.\n"
            "- context: Brief note on what this field is asking for (e.g., 'patient medication name', 'route of administration choice', 'diagnosis code')\n\n"
            "Examples:\n"
            "- Label: 'Medication:' -> type='text', context='medication name'\n"
            "- Label: 'Medication Administered: □ Self-Administered □ Physician's Office □ Other' -> type='choice', options=['Self-Administered', \"Physician's Office\", 'Other'], context='route of administration'\n"
            "- Label: 'Date of Birth:' -> type='date', context='patient DOB'\n"
            "- Label: 'ICD-10 Code(s):' -> type='code', context='diagnosis code'\n\n"
            f"FORM TEXT:\n{form_text}"
        )

        try:
            response = self._client.post(
                f"{self.settings.mistral_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.mistral_model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You analyze prior authorization forms and extract structured label metadata as strict JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = self._coerce_json(content)
            labels = parsed.get("labels", [])
            print(f"DEBUG: AI extracted {len(labels)} form labels with types")
            return labels
        except Exception as exc:
            print(f"DEBUG: AI label extraction error: {exc}")
            return []

    def extract_emr_facts(self, emr_text: str) -> dict[str, str]:
        """Extract canonical clinical facts from the EMR text."""
        canonical_fields = [
            "patient_name",
            "date_of_birth",
            "patient_id",
            "member_id",
            "diagnosis_code",
            "diagnosis_description",
            "procedure_code",
            "procedure_description",
            "medication_name",
            "covered_medication_name",
            "strength",
            "dosage",
            "frequency",
            "route_direction",
            "provider_name",
            "provider_npi",
            "office_phone",
            "office_fax",
            "patient_street_address",
            "street_address",
            "office_street_address",
            "city",
            "state",
            "zip_code",
            "insurance_id",
            "group_number",
            "specialty",
            "start_date",
            "discharge_date",
            "hospitalized",
            "pregnant",
            "allergies",
            "medical_necessity_statement",
        ]

        if not self.settings.mistral_api_key:
            return {field: "" for field in canonical_fields}

        prompt = (
            "Extract canonical clinical facts from this EMR. Return a JSON object with shape "
            "{\"facts\": {field_name: string, ...}}. Extract short, direct values only. "
            "Include patient demographics, provider info, diagnosis, procedure, medication details, "
            "insurance info, contact info, and any other clinically relevant facts. "
            "Differentiate patient_street_address vs office_street_address. If patient home address is not explicit, leave patient_street_address empty. "
            "For medications, include the medication most appropriate for PA submission and likely payer coverage as covered_medication_name when inferable from EMR context (step therapy, formulary notes, prior trial/failure, active plan).\n\n"
            f"FIELDS:\n{json.dumps(canonical_fields, indent=2)}\n\n"
            f"EMR TEXT:\n{emr_text}"
        )

        try:
            response = self._client.post(
                f"{self.settings.mistral_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.mistral_model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You extract canonical structured clinical facts from EMR text as strict JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = self._coerce_json(content)
            facts = parsed.get("facts", {})
            return {field: str(facts.get(field, "") or "") for field in canonical_fields}
        except Exception as exc:
            print(f"DEBUG: EMR fact extraction error: {exc}")
            return {field: "" for field in canonical_fields}

    def map_emr_facts_to_form_labels(self, emr_facts: dict[str, str], form_labels: list[str], label_metadata: dict[str, dict[str, Any]] | None = None) -> dict[str, str]:
        """Map canonical EMR facts onto the detected PA form labels.
        
        Args:
            emr_facts: Canonical EMR facts dictionary
            form_labels: List of form label names
            label_metadata: Optional dict of {normalized_label: {type, options, context}} from AI extraction
        """
        if not form_labels:
            return {}

        if not self.settings.mistral_api_key:
            return self._fallback_form_mapping(emr_facts, form_labels)

        label_metadata = label_metadata or {}
        
        # Enrich form labels with metadata for the AI
        enriched_labels = []
        for label in form_labels:
            normalized = self._normalize_label(label)
            metadata = label_metadata.get(normalized, {})
            label_with_meta = {
                "name": label,
                "type": metadata.get("type", "text"),
                "options": metadata.get("options", []),
                "context": metadata.get("context", ""),
            }
            enriched_labels.append(label_with_meta)

        prompt = (
            "Map canonical EMR facts to the target prior authorization form labels. Return JSON as "
            "{\"mapping\": {form_label: value, ...}}. Only include labels you can fill. "
            "Use the exact form label as the key. Values must come from the EMR facts.\n\n"
            "Default assumption: labels refer to the PATIENT unless the label text explicitly indicates provider/office/practice/facility context. "
            "If label includes words like office/practice/prescriber/provider/facility, map to provider-side facts.\n"
            "IMPORTANT: If a label is marked as type='choice', return ONLY the exact option text (e.g., 'Self-Administered'), NOT a medication name or other free text.\n"
            "For medication-related labels, prefer covered_medication_name when available; otherwise use medication_name.\n\n"
            f"TARGET FORM LABELS (with types and options):\n{json.dumps(enriched_labels, indent=2)}\n\n"
            f"EMR FACTS:\n{json.dumps(emr_facts, indent=2)}"
        )

        try:
            response = self._client.post(
                f"{self.settings.mistral_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.mistral_model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You map canonical EMR facts to exact prior authorization form labels as strict JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = self._coerce_json(content)
            mapping = parsed.get("mapping", {})
            normalized_mapping = {label: str(mapping.get(label, "") or "") for label in form_labels}
            adjusted = self._disambiguate_address_mapping(normalized_mapping, emr_facts)
            return self._semantic_fill_missing(adjusted, emr_facts, form_labels, label_metadata)
        except Exception as exc:
            print(f"DEBUG: EMR-to-form mapping error: {exc}")
            fallback = self._fallback_form_mapping(emr_facts, form_labels)
            adjusted = self._disambiguate_address_mapping(fallback, emr_facts)
            return self._semantic_fill_missing(adjusted, emr_facts, form_labels, label_metadata)

    def _semantic_fill_missing(self, mapping: dict[str, str], emr_facts: dict[str, str], form_labels: list[str], label_metadata: dict[str, dict[str, Any]] | None = None) -> dict[str, str]:
        diagnosis_text = (emr_facts.get("diagnosis_description", "") or "").strip()
        diagnosis_code = (emr_facts.get("diagnosis_code", "") or "").strip()
        route = (emr_facts.get("route_direction", "") or "").lower()
        allergies = (emr_facts.get("allergies", "") or "").strip()
        med_statement = (emr_facts.get("medical_necessity_statement", "") or "").lower()

        def infer_new_or_continuation() -> str:
            if "continu" in med_statement or "currently on" in med_statement or "maintain" in med_statement:
                return "Continuation of Therapy"
            return "New"

        def infer_med_administered() -> str:
            # Generic, form-agnostic inference by route context.
            if any(token in route for token in ["iv", "intravenous", "infusion"]):
                return "Physician's Office"
            if any(token in route for token in ["oral", "subq", "subcutaneous", "self", "po"]):
                return "Self-administered"
            return "Other"

        request_type_label = ""
        request_type_value = ""

        for label in form_labels:
            current = (mapping.get(label, "") or "").strip()
            norm = label.lower()
            if current:
                continue

            if "diagnosis for the medication" in norm:
                mapping[label] = diagnosis_text or diagnosis_code
                continue

            if "contraindication" in norm or "intolerance" in norm:
                # Map allergies to contraindication/intolerance questions if present.
                if allergies:
                    mapping[label] = allergies
                else:
                    mapping[label] = "No"
                continue

            if "is the requested medication" in norm and "continuation" in norm:
                mapping[label] = infer_new_or_continuation()
                request_type_label = label
                request_type_value = mapping[label]
                continue

            if "medication administered" in norm:
                mapping[label] = infer_med_administered()
                continue

            if "icd-10" in norm and diagnosis_code:
                mapping[label] = diagnosis_code
                continue

        # Enforce mutually exclusive New vs continuation date fields.
        request_norm = (request_type_value or "").lower()
        has_continuation_question = any(
            "continuation" in lbl.lower() and "requested medication" in lbl.lower() for lbl in form_labels
        )
        if "new" in request_norm and has_continuation_question:
            for label in form_labels:
                norm = label.lower()
                if "start date" not in norm:
                    continue
                # Clear continuation-specific start date prompts when New is selected.
                if "continuation" in norm or "requested medication" in norm or norm.strip() == "start date":
                    mapping[label] = ""

        return mapping

    def _disambiguate_address_mapping(self, mapping: dict[str, str], emr_facts: dict[str, str]) -> dict[str, str]:
        patient_address = emr_facts.get("patient_street_address", "").strip()
        office_address = emr_facts.get("office_street_address", "").strip() or emr_facts.get("street_address", "").strip()

        patient_labels = [label for label in mapping if "street address" in label.lower() and "office" not in label.lower() and "practice" not in label.lower()]
        office_labels = [label for label in mapping if "address" in label.lower() and ("office" in label.lower() or "practice" in label.lower() or "provider" in label.lower())]

        for label in office_labels:
            if office_address:
                mapping[label] = office_address

        for label in patient_labels:
            current_value = (mapping.get(label, "") or "").strip()
            if patient_address:
                mapping[label] = patient_address
                continue

            # If no explicit patient address evidence, avoid copying office address into patient address fields.
            if office_address and current_value == office_address:
                mapping[label] = ""

        return mapping

    def _fallback_form_mapping(self, emr_facts: dict[str, str], form_labels: list[str]) -> dict[str, str]:
        lower_facts = {key.lower(): value for key, value in emr_facts.items() if value}

        def pick(*candidate_keys: str) -> str:
            for candidate in candidate_keys:
                value = lower_facts.get(candidate.lower(), "")
                if value:
                    return value
            return ""

        result: dict[str, str] = {}
        for label in form_labels:
            normalized = label.lower()
            value = ""
            if "member name" in normalized or "patient name" in normalized:
                value = pick("patient_name")
            elif "date of birth" in normalized or normalized == "dob":
                value = pick("date_of_birth")
            elif "member id" in normalized or "patient id" in normalized:
                value = pick("member_id", "patient_id")
            elif "provider name" in normalized or "prescriber name" in normalized:
                value = pick("provider_name")
            elif "npi" in normalized:
                value = pick("provider_npi")
            elif "specialty" in normalized:
                value = pick("specialty")
            elif "office" in normalized and "address" in normalized:
                value = pick("office_street_address", "street_address")
            elif "practice" in normalized and "address" in normalized:
                value = pick("office_street_address", "street_address")
            elif "street address" in normalized or "address" in normalized:
                value = pick("patient_street_address", "street_address")
            elif "office phone" in normalized or normalized == "phone":
                value = pick("office_phone")
            elif "office fax" in normalized or "fax" in normalized:
                value = pick("office_fax")
            elif "diagnosis code" in normalized:
                value = pick("diagnosis_code")
            elif "diagnosis description" in normalized or "diagnosis" in normalized:
                value = pick("diagnosis_description")
            elif "procedure code" in normalized:
                value = pick("procedure_code")
            elif "procedure description" in normalized:
                value = pick("procedure_description")
            elif "medication" in normalized:
                value = pick("covered_medication_name", "medication_name")
            elif "strength" in normalized:
                value = pick("strength")
            elif "dosage" in normalized:
                value = pick("dosage")
            elif "frequency" in normalized or "direction" in normalized or "route" in normalized:
                value = pick("frequency", "route_direction")
            elif "insurance" in normalized:
                value = pick("insurance_id")
            elif "group" in normalized:
                value = pick("group_number")
            elif "allerg" in normalized:
                value = pick("allergies")
            elif "hospital" in normalized:
                value = pick("hospitalized")
            elif "pregnant" in normalized:
                value = pick("pregnant")
            elif "start date" in normalized:
                value = pick("start_date")
            elif "discharge date" in normalized:
                value = pick("discharge_date")
            elif "medical necessity" in normalized:
                value = pick("medical_necessity_statement")

            if value:
                result[label] = value

        return self._disambiguate_address_mapping(result, emr_facts)

    def extract_form_values(self, emr_text: str, form_schema: dict[str, Any]) -> MistralExtractionResult:
        if not self.settings.mistral_api_key:
            return MistralExtractionResult(raw_response="", parsed=self._fallback_extract(form_schema))

        prompt = self._build_prompt(emr_text, form_schema)
        response = self._client.post(
            f"{self.settings.mistral_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.mistral_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.mistral_model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You map clinical EMR evidence into prior authorization form fields as strict JSON."},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = self._coerce_json(content)
        return MistralExtractionResult(raw_response=content, parsed=parsed)

    def extract_generic_pa_fields(self, emr_text: str) -> MistralExtractionResult:
        """Extract common PA form fields when the form PDF has no detectable widgets."""
        if not self.settings.mistral_api_key:
            return MistralExtractionResult(raw_response="", parsed=self._fallback_generic_pa())

        prompt = (
            "Extract all relevant prior authorization information from this EMR. "
            "Return JSON with this shape: {\"fields\": [{\"name\": string, \"value\": string, \"confidence\": number, \"source\": string}], \"notes\": [string]}. "
            "Extract these fields if available: patient_name, date_of_birth, patient_id, diagnosis_code, diagnosis_description, procedure_code, procedure_description, "
            "provider_name, provider_npi, insurance_id, group_number, auth_request_date, medical_necessity_statement. "
            "Use short, direct values. If not found, use empty string and low confidence.\n\n"
            f"EMR TEXT:\n{emr_text}"
        )

        response = self._client.post(
            f"{self.settings.mistral_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.mistral_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.mistral_model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You extract prior authorization fields from clinical EMR text as strict JSON."},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = self._coerce_json(content)
        return MistralExtractionResult(raw_response=content, parsed=parsed)

    def map_ehr_to_form_fields(self, emr_text: str, form_labels: list[str]) -> dict[str, str]:
        """Match EMR data to form field labels intelligently."""
        if not self.settings.mistral_api_key or not form_labels:
            return {}

        prompt = (
            f"You are extracting clinical information from an EMR and mapping it to a prior authorization form.\n"
            f"Extract EVERY relevant piece of information from the EMR that matches ANY form field label.\n"
            f"Return a JSON object with shape: {{\"mapping\": {{label: extracted_value, ...}}}}\n"
            f"Include values for: patient demographics (name, DOB, ID), provider info, diagnosis, procedure, "
            f"medication details (name, strength, dosage, frequency, route/direction), insurance info, and any other clinical data.\n"
            f"Use exact matches when possible. Be comprehensive and include ALL extracted values that relate to the form labels.\n"
            f"Use short, direct values suitable for form fields. If a field doesn't apply, omit it.\n\n"
            f"FORM FIELD LABELS:\n{json.dumps(form_labels, indent=2)}\n\n"
            f"EMR TEXT:\n{emr_text}"
        )

        try:
            response = self._client.post(
                f"{self.settings.mistral_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.mistral_model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You are a clinical data extraction expert. Extract ALL relevant EMR information and map to form fields comprehensively."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = self._coerce_json(content)
            return parsed.get("mapping", {})
        except Exception as e:
            print(f"DEBUG: EHR-to-form mapping error: {e}")
            return {}

    def suggest_field_mapping(self, extracted_field_names: list[str], form_widget_names: list[str], form_name: str = "") -> dict[str, str]:
        """Suggest a mapping from extracted PA field names to actual form widget names."""
        if not self.settings.mistral_api_key or not form_widget_names:
            return {}

        prompt = (
            f"Map the following extracted prior authorization field names to the actual form widget field names.\n"
            f"Return a JSON object with shape: {{\"mapping\": {{extracted_name: widget_name, ...}}, \"notes\": [string]}}.\n"
            f"Only include mappings where the semantics match clearly. Leave out fields you cannot map.\n\n"
            f"EXTRACTED PA FIELDS:\n{json.dumps(extracted_field_names, indent=2)}\n\n"
            f"FORM WIDGET NAMES:\n{json.dumps(form_widget_names, indent=2)}\n\n"
            f"Form name/context: {form_name if form_name else 'Unknown'}"
        )

        try:
            response = self._client.post(
                f"{self.settings.mistral_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.mistral_model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You are a field mapping expert. Map extracted data field names to form widget field names semantically."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = self._coerce_json(content)
            return parsed.get("mapping", {})
        except Exception as e:
            print(f"DEBUG: Field mapping error: {e}")
            return {}

    def _build_prompt(self, emr_text: str, form_schema: dict[str, Any]) -> str:
        fields_json = json.dumps(form_schema, indent=2)
        return (
            "Extract the best possible field values from this EMR for the following prior authorization form schema.\n"
            "Return JSON with this shape: {\"fields\": [{\"name\": string, \"value\": string, \"confidence\": number, \"source\": string}], \"notes\": [string]}.\n"
            "Only include fields that exist in the schema. Use short, direct values. If a field is not supported by the EMR, use an empty string and low confidence.\n\n"
            f"FORM SCHEMA:\n{fields_json}\n\nEMR TEXT:\n{emr_text}"
        )

    def _coerce_json(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
            return self._fallback_extract({})

    def _fallback_extract(self, form_schema: dict[str, Any]) -> dict[str, Any]:
        fields = []
        for field in form_schema.get("fields", []):
            fields.append({"name": field.get("name", ""), "value": "", "confidence": 0.0, "source": "fallback"})
        return {"fields": fields, "notes": ["Mistral API key not configured, using empty fallback values."]}

    def _fallback_generic_pa(self) -> dict[str, Any]:
        default_fields = [
            "patient_name", "date_of_birth", "patient_id", "diagnosis_code", "diagnosis_description",
            "procedure_code", "procedure_description", "provider_name", "provider_npi",
            "insurance_id", "group_number", "auth_request_date", "medical_necessity_statement"
        ]
        fields = [{"name": fname, "value": "", "confidence": 0.0, "source": "fallback"} for fname in default_fields]
        return {"fields": fields, "notes": ["Mistral API key not configured, using empty fallback values."]}
