from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


class PdfOverlay:
    """Detect labels in text-based PDFs and place values beside them."""

    def __init__(self) -> None:
        self.label_metadata: dict[str, dict[str, Any]] = {}  # Store label type/options from AI

    def extract_form_labels(self, template_pdf_path: str | Path) -> list[str]:
        # For now, extract raw text and delegate to AI extraction via mistral
        document = fitz.open(str(template_pdf_path))
        try:
            form_text = ""
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = page.get_text("text") or ""
                form_text += text + "\n"
            
            # Import here to avoid circular dependency
            from app.services.mistral_client import MistralClient
            mistral = MistralClient()
            try:
                ai_labels = mistral.extract_form_labels_with_ai(form_text)
                
                # Store metadata and return just the names
                labels: list[str] = []
                for label_info in ai_labels:
                    name = label_info.get("name", "").strip()
                    if name:
                        labels.append(name)
                        # Store metadata for later use in overlay logic
                        normalized = self._normalize_label(name)
                        self.label_metadata[normalized] = {
                            "type": label_info.get("type", "text"),
                            "options": label_info.get("options", []),
                            "context": label_info.get("context", ""),
                        }
                
                # Contextualize labels using page coordinates and nearby section headers.
                labels = self._contextualize_duplicate_labels_simple(labels, document)
                
                # Remap metadata to use contextualized labels
                old_metadata = self.label_metadata
                self.label_metadata = {}
                for contextual_label in labels:
                    base_label = self._extract_base_label(contextual_label)
                    normalized_base = self._normalize_label(base_label)
                    if normalized_base in old_metadata:
                        self.label_metadata[self._normalize_label(contextual_label)] = old_metadata[normalized_base]
                
                print(f"DEBUG: extract_form_labels() found {len(labels)} labels (after contextualization) = {labels[:10]}...")
                return labels
            finally:
                mistral.close()
        finally:
            document.close()

    def _deduplicate_substring_labels(self, labels: list[str]) -> list[str]:
        """Remove shorter labels whose normalized form is a substring of longer labels.
        E.g., keep 'Directions for use' and remove 'Direction'."""
        if not labels:
            return labels
        
        # Sort by normalized length (descending) to process longer ones first
        sorted_labels = sorted(labels, key=lambda l: len(self._normalize_label(l)), reverse=True)
        kept: list[str] = []
        
        for label in sorted_labels:
            label_norm = self._normalize_label(label)
            # Check if this label's normalized form is a substring of any already-kept label
            is_substring = False
            for kept_label in kept:
                kept_norm = self._normalize_label(kept_label)
                if label_norm != kept_norm and label_norm in kept_norm:
                    is_substring = True
                    print(f"DEBUG: Removing substring label '{label}' (normalized='{label_norm}') as it's contained in '{kept_label}' (normalized='{kept_norm}')")
                    break
            
            if not is_substring:
                kept.append(label)
        
        return kept

    def _contextualize_duplicate_labels_simple(self, labels: list[str], document: fitz.Document) -> list[str]:
        """Contextualize labels using page coordinates instead of raw occurrence counting.

        The goal is to preserve the same base label while attaching a stable section hint
        derived from layout: nearby section header, column side, or vertical region.
        For duplicate base labels, different occurrences get different hints based on position.
        """
        if not labels:
            return labels

        # First pass: collect ALL occurrences of ALL unique base labels from the PDF
        occurrences: dict[str, list[tuple[int, fitz.Rect, str]]] = {}
        unique_base_labels = set()
        for label in labels:
            base_label = self._extract_base_label(label)
            unique_base_labels.add(base_label)

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            for base_label in unique_base_labels:
                for rect in self._find_label_rects(page, base_label):
                    section_hint = self._infer_section_hint(page, rect)
                    occurrences.setdefault(self._normalize_label(base_label), []).append((page_index, rect, section_hint))

        # Sort occurrences by y0 to ensure consistent top-to-bottom ordering
        for normalized in occurrences:
            occurrences[normalized].sort(key=lambda t: (t[0], t[1].y0, t[1].x0))

        # Second pass: contextualize each input label based on its occurrence index
        contextualized: list[str] = []
        seen_counts: dict[str, int] = {}

        for label in labels:
            base_label = self._extract_base_label(label)
            normalized = self._normalize_label(base_label)
            seen_counts[normalized] = seen_counts.get(normalized, 0) + 1
            occurrence_index = seen_counts[normalized] - 1

            section_hint = ""
            candidate_list = occurrences.get(normalized, [])
            
            # If we found multiple occurrences of this base label with different hints, use them
            if occurrence_index < len(candidate_list):
                section_hint = candidate_list[occurrence_index][2]
            elif len(candidate_list) > 0:
                # If we only found one occurrence but this is a duplicate label request,
                # use position-based heuristic: first on left, second on right
                fallback_hint = "Right Column" if seen_counts[normalized] > 1 else candidate_list[0][2]
                section_hint = fallback_hint
            
            if not section_hint:
                section_hint = f"Occurrence {seen_counts[normalized]}"

            contextual_label = f"{base_label} [{section_hint}]"
            contextualized.append(contextual_label)

        return contextualized

    def _find_label_rects(self, page: fitz.Page, label: str) -> list[fitz.Rect]:
        """Return candidate rectangles for a label on a page using exact and variant searches."""
        candidates: list[fitz.Rect] = []
        for variant in self._label_variants(label):
            search_results = page.search_for(variant)
            if search_results:
                candidates.extend(search_results)
        # De-duplicate rectangles more aggressively to avoid near-duplicates
        unique: list[fitz.Rect] = []
        seen = set()
        for rect in sorted(candidates, key=lambda r: (r.y0, r.x0)):
            # Use tighter threshold for deduplication: if center points are within 5 pixels, consider them duplicates
            key = (round(rect.x0 / 5) * 5, round(rect.y0 / 5) * 5)
            if key not in seen:
                seen.add(key)
                unique.append(rect)
        return sorted(unique, key=lambda rect: (rect.y0, rect.x0))

    def _infer_section_hint(self, page: fitz.Page, label_rect: fitz.Rect) -> str:
        """Infer a section hint from nearby headers or page geometry.
        
        Priority: Section headers > Layout-based hints
        """
        layout = self._detect_layout(page)
        label_center_x = (label_rect.x0 + label_rect.x1) / 2
        label_center_y = (label_rect.y0 + label_rect.y1) / 2

        if layout == "two_column_vertical":
            return "Left Column" if label_center_x < (page.rect.width / 2) else "Right Column"

        # For horizontal or single-column layouts, try headers first so stacked sections
        # can be labeled by section name instead of a generic band.
        header = self._nearest_section_header(page, label_rect)
        if header:
            return header

        if layout == "two_row_horizontal":
            return "Top Section" if label_center_y < (page.rect.height / 2) else "Bottom Section"

        # Single column fallback
        return "Upper Section" if label_center_y < (page.rect.height / 2) else "Lower Section"

    def _detect_layout(self, page: fitz.Page) -> str:
        """Detect whether the page is predominantly two-column, two-row, or single-column."""
        words = page.get_text("words") or []
        if len(words) < 8:
            return "single_column"

        midpoint_x = page.rect.width / 2
        midpoint_y = page.rect.height / 2
        x_centers = [((word[0] + word[2]) / 2) for word in words if str(word[4]).strip()]
        y_centers = [((word[1] + word[3]) / 2) for word in words if str(word[4]).strip()]

        left = sum(1 for x in x_centers if x < midpoint_x)
        right = sum(1 for x in x_centers if x >= midpoint_x)
        upper = sum(1 for y in y_centers if y < midpoint_y)
        lower = sum(1 for y in y_centers if y >= midpoint_y)

        # A vertical two-column form usually has a noticeable gap around the page midpoint.
        gap_x = sum(1 for x in x_centers if abs(x - midpoint_x) < 45)
        left_density = left / max(len(x_centers), 1)
        right_density = right / max(len(x_centers), 1)
        gap_ratio = gap_x / max(len(x_centers), 1)

        # Horizontal split fallback for stacked sections.
        gap_y = sum(1 for y in y_centers if abs(y - midpoint_y) < 45)
        upper_density = upper / max(len(y_centers), 1)
        lower_density = lower / max(len(y_centers), 1)
        gap_y_ratio = gap_y / max(len(y_centers), 1)

        # Prefer horizontal sections only when top/bottom separation is very strong.
        horizontal_strong = (
            upper > 0
            and lower > 0
            and gap_y_ratio < 0.09
            and (upper_density > 0.28 and lower_density > 0.28)
        )
        if horizontal_strong:
            return "two_row_horizontal"

        if left > 0 and right > 0 and gap_ratio < 0.25:
            return "two_column_vertical"

        if upper > 0 and lower > 0 and gap_y_ratio < 0.20 and (upper_density > 0.22 or lower_density > 0.22):
            return "two_row_horizontal"
        return "single_column"

    def _nearest_section_header(self, page: fitz.Page, label_rect: fitz.Rect) -> str:
        """Find the nearest recognized section header above a label.
        
        Uses flexible pattern matching to find headers like:
        - "Patient Information", "Member Information", "Member Data"
        - "Prescriber Information", "Provider Information", "Provider Data"
        - "Pharmacy Information", "Dispensing Pharmacy"
        
        Returns header classification like "Patient Information" or "Prescriber Information" (not hardcoded section names).
        """
        label_center_x = (label_rect.x0 + label_rect.x1) / 2
        page_mid_x = page.rect.width / 2
        
        # Keywords that indicate patient/member section
        patient_keywords = {"patient", "member", "enrollee", "subscriber"}
        # Keywords that indicate prescriber/provider section
        provider_keywords = {"prescriber", "provider", "physician", "doctor", "office", "practice"}
        # Keywords that indicate pharmacy section
        pharmacy_keywords = {"pharmacy", "dispensing"}
        
        best_role = ""
        best_score = None

        # Evaluate header candidates line-by-line so multi-line blocks can still expose
        # clean section headers like "Patient Information" and "Prescriber Information".
        for block in page.get_text("blocks") or []:
            if len(block) < 5:
                continue

            x0, y0, x1, y1, raw_text = block[0], block[1], block[2], block[3], str(block[4] or "")
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if not lines:
                continue

            block_height = max(1.0, float(y1 - y0))
            line_height = block_height / max(len(lines), 1)

            for index, text_stripped in enumerate(lines):
                # Reject obvious field-label lines; section headers are usually concise.
                if len(text_stripped) < 3 or len(text_stripped) > 80:
                    continue
                if ":" in text_stripped or "?" in text_stripped:
                    continue

                text_normalized = self._normalize_label(text_stripped).lower()
                if not text_normalized:
                    continue

                # Approximate a line-level rectangle within the original block.
                line_y0 = y0 + (index * line_height)
                line_y1 = min(y1, line_y0 + line_height)

                # Header should appear above the label.
                if line_y1 > label_rect.y0 + 8:
                    continue

                role = ""
                if any(kw in text_normalized for kw in patient_keywords):
                    role = "Patient Information"
                elif any(kw in text_normalized for kw in provider_keywords):
                    role = "Prescriber Information"
                elif any(kw in text_normalized for kw in pharmacy_keywords):
                    role = "Pharmacy Information"
                if not role:
                    continue

                # Header-like guardrails to avoid matching normal questions/labels.
                has_information_word = "information" in text_normalized
                compact_header = len(text_stripped.split()) <= 4
                is_header_like = has_information_word or text_stripped.isupper() or compact_header
                if not is_header_like:
                    continue

                vertical_distance = label_rect.y0 - line_y1
                horizontal_distance = abs(((x0 + x1) / 2) - label_center_x)

                # Allow larger distances for stacked top/bottom sections.
                if vertical_distance > 260:
                    continue

                # Wide header bands (often in horizontal layouts) should not be penalized
                # by side mismatch. Narrow headers still prefer same-side matches.
                header_width_ratio = (x1 - x0) / max(1.0, float(page.rect.width))
                same_side_penalty = 0
                if header_width_ratio < 0.35:
                    same_side_penalty = 0 if (((x0 + x1) / 2 < page_mid_x) == (label_center_x < page_mid_x)) else 20

                score = (vertical_distance * 0.8) + (horizontal_distance * 0.2) + same_side_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    best_role = role

        return best_role

    def _extract_base_label(self, label: str) -> str:
        """Extract the base label from a contextualized label.
        
        Strips both section hints like '[Upper Section]' and legacy occurrence markers
        like '(occurrence 2)'.
        """
        cleaned = re.sub(r"\s*\[(.+)\]\s*$", "", label.strip())
        match = re.match(r"^(.*?)\s*\(occurrence\s+\d+\)\s*$", cleaned)
        if match:
            return match.group(1).strip()
        return cleaned.strip()

    def _extract_occurrence_number(self, label: str) -> int:
        """Extract the 1-based occurrence number from a contextualized label.

        Returns 1 when the label has no explicit occurrence suffix.
        """
        cleaned = re.sub(r"\s*\[(.+)\]\s*$", "", label.strip())
        match = re.match(r"^.*\(occurrence\s+(\d+)\)\s*$", cleaned, flags=re.IGNORECASE)
        if match:
            try:
                return max(1, int(match.group(1)))
            except ValueError:
                return 1
        return 1

    def fill_form_with_overlay(
        self, template_pdf_path: str | Path, field_values: dict[str, str], output_pdf_path: str | Path
    ) -> str:
        document = fitz.open(str(template_pdf_path))
        try:
            label_positions = self._find_label_positions_for_labels(document, list(field_values.keys()))
            print(f"DEBUG: found label_positions with {len(label_positions)} entries")

            for label, value in field_values.items():
                if not label or not value:
                    continue

                rect = label_positions.get(label)  # Look up by full contextualized label
                if not rect:
                    print(f"DEBUG: No position found for label '{label}'")
                    continue

                page_index, rect_obj = rect
                page = document.load_page(page_index)

                # Get AI-determined label type (use contextualized label as key)
                label_type = self.label_metadata.get(self._normalize_label(label), {}).get("type", "text")
                label_options = self.label_metadata.get(self._normalize_label(label), {}).get("options", [])

                # If AI says this is a choice field, try choice marking first
                if label_type == "choice" and label_options:
                    if self._mark_choice_if_possible(page, rect_obj, label, str(value), label_options):
                        print(f"DEBUG: Successfully marked choice for '{label[:50]}'")
                        continue

                if self._should_skip_duplicate_overlay(page, rect_obj, str(value)):
                    print(f"DEBUG: Skipping duplicate overlay for label='{label}' value='{value}'")
                    continue

                # For long free-text answers, write what fits on the right first,
                # then continue wrapping on the next line(s) below.
                if self._render_right_then_below_text(page, rect_obj, label, str(value), fontsize=10):
                    continue

                x_offset, y_pos = self._compute_text_position(page, rect_obj, label, str(value))
                print(f"DEBUG: Overlaying label='{label}' value='{value}' at ({x_offset}, {y_pos})")

                page.insert_text(
                    (x_offset, y_pos),
                    value,
                    fontsize=10,
                    color=(0, 0, 0),
                )

            output_path = Path(output_pdf_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            document.save(str(output_path), incremental=False, deflate=True, garbage=4)
            return str(output_path)
        finally:
            document.close()

    def _find_label_positions_for_labels(
        self, document: fitz.Document, labels_to_find: list[str]
    ) -> dict[str, tuple[int, fitz.Rect]]:
        """Find positions for all labels, including contextualized duplicates like 'City [Left Column]' and 'City [Right Column]'.
        
        For duplicate base labels (same label appearing multiple times with different section hints),
        searches for the Nth occurrence in sorted order (top-to-bottom, left-to-right).
        """
        positions: dict[str, tuple[int, fitz.Rect]] = {}
        
        # Track how many times we've searched for each base label
        base_label_search_count: dict[str, int] = {}
        
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_text = page.get_text("text") or ""
            
            for label in labels_to_find:
                if label in positions:
                    continue  # Skip if THIS specific contextualized label already found

                base_label = self._extract_base_label(label)
                
                # Track which occurrence of this base label we're looking for
                base_label_search_count[base_label] = base_label_search_count.get(base_label, 0) + 1
                occurrence_number = base_label_search_count[base_label]
                
                label_variants = self._label_variants(base_label)
                found_rect = None
                search_results_for_label: list[fitz.Rect] = []
                
                for variant in label_variants:
                    search_results = page.search_for(variant)
                    if search_results:
                        search_results_for_label = search_results
                        break

                if search_results_for_label:
                    ranked_candidates = sorted(search_results_for_label, key=lambda rect: (rect.y0, rect.x0))
                    candidate_index = min(occurrence_number - 1, len(ranked_candidates) - 1)
                    found_rect = ranked_candidates[candidate_index]

                if found_rect is None:
                    # Fallback: look for the label text in the extracted page text lines.
                    for line in page_text.splitlines():
                        line_normalized = self._normalize_label(line)
                        base_label_norm = self._normalize_label(base_label)
                        # Only use substring matching for longer text or exact matches to avoid collisions
                        if line_normalized == base_label_norm or (
                            len(base_label_norm) > 20 and base_label_norm in line_normalized
                        ):
                            search_results = page.search_for(line.strip())
                            if search_results:
                                ranked_candidates = sorted(search_results, key=lambda rect: (rect.y0, rect.x0))
                                candidate_index = min(occurrence_number - 1, len(ranked_candidates) - 1)
                                found_rect = ranked_candidates[candidate_index]
                            break

                if found_rect is not None:
                    positions[label] = (page_index, found_rect)  # Key by contextualized label
                    print(f"DEBUG: Found position for label '{label}' at {found_rect}")

        return positions

    def _labels_from_text(self, text: str) -> list[str]:
        label_patterns = [
            "Member Name",
            "Patient Name",
            "Date Of Birth",
            "Date of Birth",
            "Member ID",
            "Patient ID",
            "Street Address",
            "Office Address",
            "Office Phone",
            "Office Fax",
            "Provider Name",
            "Prescriber Name",
            "NPI #",
            "NPI Number",
            "Specialty",
            "Diagnosis Code",
            "Diagnosis Description",
            "Procedure Code",
            "Procedure Description",
            "Medication",
            "Medication Name",
            "Strength",
            "Dosage",
            "Frequency",
            "Direction",
            "Route",
            "Insurance ID",
            "Group Number",
            "Group #",
            "Authorization Date",
            "Medical Necessity",
            "Hospitalized",
            "Pregnant",
            "Start Date",
            "Discharge Date",
            "Allergies",
        ]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        labels: list[str] = []
        seen: set[str] = set()

        def add_label(candidate: str) -> None:
            normalized = self._normalize_label(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                labels.append(candidate)

        for pattern in label_patterns:
            if pattern.lower() in text.lower():
                add_label(pattern)

        for line in lines:
            compact = re.sub(r"\s+", " ", line).strip()

            if ":" in compact:
                candidate = compact.split(":", 1)[0].strip()
                if 1 <= len(candidate) <= 180:
                    add_label(candidate)

            # Capture question clauses even when the line has extra trailing text.
            for chunk in re.findall(r"([^?.!]{8,260}\?)", compact):
                candidate = re.sub(r"\s+", " ", chunk).strip()
                if candidate:
                    add_label(candidate)

            # Capture complete question-style labels used in many PA forms.
            if compact.endswith("?"):
                candidate = compact.strip()
                if 8 <= len(candidate) <= 260:
                    add_label(candidate)

            # Some forms include prompt labels without trailing question marks.
            if not compact.endswith("?") and self._looks_like_prompt_label(compact):
                add_label(compact.rstrip(":"))

            # Capture sentence labels with trailing colon at full-line granularity.
            if compact.endswith(":"):
                candidate = compact.rstrip(":").strip()
                if 3 <= len(candidate) <= 220:
                    add_label(candidate)

        return labels

    def _label_variants(self, label: str) -> list[str]:
        base = label.strip().rstrip(":")
        normalized = self._normalize_label(base)
        variants = [base, f"{base}:"]

        words = [part for part in re.split(r"\s+", base) if part]
        if len(words) >= 5:
            variants.extend(
                [
                    " ".join(words[:6]),
                    " ".join(words[-6:]),
                ]
            )

        if normalized == "dateofbirth":
            variants.extend(["Date Of Birth", "Date of Birth", "DOB"])
        elif normalized == "membername":
            variants.append("Member Name")
        elif normalized == "providername":
            variants.append("Provider Name")
        elif normalized == "npi":
            variants.extend(["NPI", "NPI #", "NPI Number"])

        return list(dict.fromkeys(variants))

    def _choose_best_label_rect(self, page: fitz.Page, label: str, candidates: list[fitz.Rect]) -> fitz.Rect:
        def score(rect: fitz.Rect) -> tuple[int, int, int, float]:
            line_text = self._line_text_near_label(page, rect).lower()
            label_l = label.lower().strip().rstrip(":")
            label_norm = self._normalize_label(label_l)
            matched_text = (page.get_textbox(rect) or "").strip().lower()
            matched_norm = self._normalize_label(matched_text)

            exact_prefix = 1 if line_text.strip().startswith(label_l) else 0
            has_colon = 1 if ":" in line_text else 0
            is_question = 1 if "?" in line_text else 0
            exact_token = 1 if matched_norm == label_norm else 0

            # Penalize cases where short labels appear inside long question prompts.
            generic_label = label_l in {"medication", "direction", "phone", "state", "city"}
            question_penalty = 1 if (generic_label and is_question) else 0

            # Penalize substring hits for one-word labels (e.g., "Direction" in "Directions").
            partial_word_penalty = 1 if (len(label_l.split()) == 1 and matched_norm and matched_norm != label_norm) else 0

            return (
                exact_prefix + has_colon + exact_token - question_penalty - partial_word_penalty,
                -len(line_text),
                -len(matched_text),
                rect.x0,
            )

        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[0]

    def _normalize_label(self, label: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", label.lower())

    def _right_anchor_after_immediate_content(self, page: fitz.Page, label_rect: fitz.Rect) -> float:
        """Return x anchor just after content immediately following the label on same row.

        This captures contiguous inline text (e.g., parenthetical instructions) but avoids
        jumping across big gaps into other columns.
        """
        label_center_y = (label_rect.y0 + label_rect.y1) / 2
        y_tolerance = 4.5
        gap_threshold = 12.0
        anchor_x = label_rect.x1

        row_words = []
        for word in page.get_text("words") or []:
            x0, y0, x1, y1, text, *_ = word
            if not re.search(r"[A-Za-z0-9]", str(text or "")):
                continue
            word_center_y = (y0 + y1) / 2
            if abs(word_center_y - label_center_y) > y_tolerance:
                continue
            if x1 <= label_rect.x1 + 0.5:
                continue
            row_words.append((x0, x1))

        row_words.sort(key=lambda item: item[0])
        for x0, x1 in row_words:
            # Extend only through immediately adjacent/contiguous text fragments.
            if x0 <= anchor_x + gap_threshold:
                anchor_x = max(anchor_x, x1)
                continue
            break

        return min(anchor_x + 4, page.rect.width - 20)

    def _compute_text_position(self, page: fitz.Page, rect_obj: fitz.Rect, label: str, value: str) -> tuple[float, float]:
        right_x = self._right_anchor_after_immediate_content(page, rect_obj)
        center_y = (rect_obj.y0 + rect_obj.y1) / 2 + 3
        text_width = self._estimate_text_width(value, fontsize=10)
        right_edge = page.rect.width - 6
        remaining_right_space = page.rect.width - right_x

        # Always try right placement first for ANY label
        if right_x + text_width <= right_edge and not self._would_overlap_existing_text(
            page, right_x, center_y, text_width, exclude_rect=rect_obj, y_tolerance=4.0
        ):
            print(f"DEBUG: label='{label[:50]}' strategy=RIGHT (x={right_x:.1f}, y={center_y:.1f})")
            return (right_x, center_y)

        # Try left inline placement only if there's real space on the left
        left_x = max(rect_obj.x0 - text_width - 5, 6)
        if left_x > 6 and left_x + text_width <= right_edge and not self._would_overlap_existing_text(
            page, left_x, center_y, text_width, exclude_rect=rect_obj, y_tolerance=4.0
        ):
            print(f"DEBUG: label='{label[:50]}' strategy=LEFT-INLINE (x={left_x:.1f}, y={center_y:.1f})")
            return (left_x, center_y)

        # Last resort: place below the label with minimal offset
        x = min(max(right_x, 6), max(page.rect.width - text_width - 6, 6))
        y = rect_obj.y1 + 7  # Just 7 pixels below label bottom to minimize overlap with next field
        
        # If still overlapping, try moving further down
        if self._would_overlap_existing_text(page, x, y, text_width, exclude_rect=rect_obj, y_tolerance=6.0):
            print("moving further")
            y = min(rect_obj.y1 + 16, page.rect.height - 8)
        
        print(f"DEBUG: label='{label[:50]}' strategy=BELOW-LINE (x={x:.1f}, y={y:.1f})")
        return (x, y)

    def _mark_choice_if_possible(self, page: fitz.Page, label_rect: fitz.Rect, label: str, value: str, ai_options: list[str] | None = None) -> bool:
        chosen = value.strip()
        if not chosen:
            return False

        line_text = self._line_text_near_label(page, label_rect)
        
        # Use AI-provided options if available, otherwise extract from line
        if ai_options:
            options = ai_options
        else:
            if not self._is_choice_context(label, line_text):
                print(f"DEBUG: choice_context check failed for label='{label[:50]}' line_text='{line_text[:60]}'")
                return False
            options = self._extract_options_from_line(line_text)
        
        if not options:
            print(f"DEBUG: no options available for label='{label[:50]}'")
            return False

        chosen_norm = self._normalize_label(chosen)
        selected_option = ""
        for option in options:
            option_norm = self._normalize_label(option)
            if chosen_norm == option_norm or chosen_norm in option_norm or option_norm in chosen_norm:
                selected_option = option
                break

        if not selected_option:
            print(f"DEBUG: chosen value '{chosen}' did not match any option in {options}")
            return False

        option_rect = self._find_token_on_label_line(page, selected_option, label_rect)
        if option_rect is None:
            print(f"DEBUG: option '{selected_option}' not found on label line for '{label[:50]}'")
            return False

        check_x = max(option_rect.x0 - 8, 6)
        check_y = (option_rect.y0 + option_rect.y1) / 2 + 3
        page.insert_text((check_x, check_y), "X", fontsize=10, color=(0, 0, 0))

        # If there is free text after "other:", place it after the option text.
        if chosen.lower().startswith("other") and ":" in chosen:
            other_value = chosen.split(":", 1)[1].strip()
            if other_value:
                page.insert_text((option_rect.x1 + 6, check_y), other_value, fontsize=10, color=(0, 0, 0))
        print(f"DEBUG: Marked choice label='{label[:50]}' value='{chosen}' using option '{selected_option}'")
        return True

    def _looks_like_prompt_label(self, text: str) -> bool:
        lowered = text.lower().strip()
        if len(lowered) < 20 or len(lowered) > 260:
            return False
        if lowered.endswith(":"):
            return False
        starters = (
            "what ",
            "why ",
            "when ",
            "where ",
            "which ",
            "who ",
            "how ",
            "is ",
            "are ",
            "does ",
            "do ",
            "did ",
            "has ",
            "have ",
            "can ",
            "could ",
            "should ",
            "will ",
            "would ",
        )
        return lowered.startswith(starters) and len(lowered.split()) >= 5

    def _estimate_text_width(self, value: str, fontsize: float) -> float:
        if not value:
            return 24.0
        try:
            return float(fitz.get_text_length(value, fontsize=fontsize)) + 2.0
        except Exception:
            return max(24.0, len(value) * fontsize * 0.52)

    def _split_text_to_width(self, text: str, max_width: float, fontsize: float) -> tuple[str, str]:
        stripped = " ".join(text.split())
        if not stripped:
            return ("", "")

        if self._estimate_text_width(stripped, fontsize) <= max_width:
            return (stripped, "")

        words = stripped.split(" ")
        head_words: list[str] = []
        for idx, word in enumerate(words):
            candidate = " ".join(head_words + [word]).strip()
            if candidate and self._estimate_text_width(candidate, fontsize) <= max_width:
                head_words.append(word)
                continue

            if not head_words:
                # Force at least one token to prevent infinite retry on very narrow widths.
                head_words.append(word)
                remainder = " ".join(words[idx + 1 :]).strip()
                return (" ".join(head_words).strip(), remainder)

            remainder = " ".join(words[idx:]).strip()
            return (" ".join(head_words).strip(), remainder)

        return (" ".join(head_words).strip(), "")

    def _render_right_then_below_text(
        self,
        page: fitz.Page,
        rect_obj: fitz.Rect,
        label: str,
        value: str,
        fontsize: float = 10,
    ) -> bool:
        value_clean = " ".join(value.split()).strip()
        if not value_clean:
            return False

        right_x = self._right_anchor_after_immediate_content(page, rect_obj)
        center_y = (rect_obj.y0 + rect_obj.y1) / 2 + 3
        right_edge = page.rect.width - 6
        available_right_width = right_edge - right_x

        # Only activate this mode when we can place at least one token on the right,
        # and the full value does not fit as a single line there.
        if available_right_width < 30:
            return False
        if self._estimate_text_width(value_clean, fontsize) <= available_right_width:
            return False

        first_line, remainder = self._split_text_to_width(value_clean, available_right_width, fontsize)
        if not first_line:
            return False

        # If only a tiny fragment fits on the right, skip split rendering and
        # render the full value below as a wrapped block.
        first_words = len(first_line.split())
        first_ratio = len(first_line) / max(len(value_clean), 1)
        if first_words <= 1 or first_ratio < 0.25:
            return self._render_below_wrapped_text(page, rect_obj, label, value_clean, fontsize)

        first_width = self._estimate_text_width(first_line, fontsize)
        if self._would_overlap_existing_text(
            page,
            right_x,
            center_y,
            first_width,
            exclude_rect=rect_obj,
            y_tolerance=4.0,
        ):
            return False

        page.insert_text((right_x, center_y), first_line, fontsize=fontsize, color=(0, 0, 0))

        if not remainder:
            print(
                f"DEBUG: label='{label[:50]}' strategy=RIGHT-FLOW (single-line x={right_x:.1f}, y={center_y:.1f})"
            )
            return True

        # Continue remaining content below, starting from where the label starts.
        line_height = fontsize + 2
        below_x = max(rect_obj.x0 + 2, 6)
        below_width = max(page.rect.width - 6 - below_x, 30)
        below_y = min(rect_obj.y1 + 7, page.rect.height - 8)
        max_lines_below = 3
        lines_written = 0
        tail = remainder

        while tail and lines_written < max_lines_below:
            line_text, tail = self._split_text_to_width(tail, below_width, fontsize)
            if not line_text:
                break

            line_y = below_y + (lines_written * line_height)
            if line_y > page.rect.height - 8:
                break

            # If target line is occupied, drop slightly for this continuation line.
            if self._would_overlap_existing_text(
                page,
                below_x,
                line_y,
                self._estimate_text_width(line_text, fontsize),
                exclude_rect=rect_obj,
                y_tolerance=6.0,
            ):
                line_y = min(line_y + 18, page.rect.height - 8)

            page.insert_text((below_x, line_y), line_text, fontsize=fontsize, color=(0, 0, 0))
            lines_written += 1

        print(
            f"DEBUG: label='{label[:50]}' strategy=RIGHT-THEN-BELOW (first_x={right_x:.1f}, first_y={center_y:.1f}, below_x={below_x:.1f}, below_lines={lines_written})"
        )
        return True

    def _render_below_wrapped_text(
        self,
        page: fitz.Page,
        rect_obj: fitz.Rect,
        label: str,
        value: str,
        fontsize: float = 10,
    ) -> bool:
        below_x = max(rect_obj.x0 + 2, 6)
        below_width = max(page.rect.width - 6 - below_x, 30)
        below_y = min(rect_obj.y1 + 7, page.rect.height - 8)
        line_height = fontsize + 2
        max_lines = 4
        lines_written = 0
        tail = value

        while tail and lines_written < max_lines:
            line_text, tail = self._split_text_to_width(tail, below_width, fontsize)
            if not line_text:
                break

            line_y = below_y + (lines_written * line_height)
            if line_y > page.rect.height - 8:
                break

            if self._would_overlap_existing_text(
                page,
                below_x,
                line_y,
                self._estimate_text_width(line_text, fontsize),
                exclude_rect=rect_obj,
                y_tolerance=6.0,
            ):
                line_y = min(line_y + 20, page.rect.height - 8)

            page.insert_text((below_x, line_y), line_text, fontsize=fontsize, color=(0, 0, 0))
            lines_written += 1

        print(
            f"DEBUG: label='{label[:50]}' strategy=BELOW-WRAPPED (below_x={below_x:.1f}, below_lines={lines_written})"
        )
        return lines_written > 0

    def _would_overlap_existing_text(
        self,
        page: fitz.Page,
        x: float,
        baseline_y: float,
        width: float,
        exclude_rect: fitz.Rect | None = None,
        y_tolerance: float = 8.0,
    ) -> bool:
        text_box = fitz.Rect(x, baseline_y - 9, x + width, baseline_y + 3)
        words = page.get_text("words") or []
        for word in words:
            x0, y0, x1, y1, _text, *_ = word
            word_center_y = (y0 + y1) / 2
            if abs(word_center_y - baseline_y) > y_tolerance:
                continue

            # Ignore pure punctuation/line-art tokens when checking overlap.
            if not re.search(r"[A-Za-z0-9]", str(_text or "")):
                continue

            word_rect = fitz.Rect(x0, y0, x1, y1)
            if exclude_rect is not None and word_rect.intersects(exclude_rect):
                continue
            if text_box.intersects(word_rect):
                return True
        return False

    def _should_skip_duplicate_overlay(self, page: fitz.Page, label_rect: fitz.Rect, value: str) -> bool:
        normalized_value = self._normalize_label(value)
        if not normalized_value:
            return False

        # Only apply duplicate guard to checkbox/choice keywords to avoid blocking text overlays.
        choice_keywords = {"yes", "no", "other", "new", "continuation"}
        if normalized_value not in choice_keywords:
            return False

        line_text = self._line_text_near_label(page, label_rect)
        return normalized_value in self._normalize_label(line_text)

    def _line_text_near_label(self, page: fitz.Page, label_rect: fitz.Rect) -> str:
        words = page.get_text("words") or []
        label_center_y = (label_rect.y0 + label_rect.y1) / 2
        band_min = label_center_y - 8
        band_max = label_center_y + 8

        line_words = []
        for word in words:
            x0, y0, x1, y1, text, *_ = word
            word_center_y = (y0 + y1) / 2
            if band_min <= word_center_y <= band_max:
                line_words.append((x0, text))

        line_words.sort(key=lambda item: item[0])
        return " ".join(text for _, text in line_words)

    def _is_choice_context(self, label: str, line_text: str) -> bool:
        label_l = label.lower()
        line_l = line_text.lower()
        if "□" in line_text:
            return True
        if " or " in line_l:
            return True
        if " yes " in f" {line_l} " and " no " in f" {line_l} ":
            return True
        has_choice_token = any(token in label_l for token in ["administered", "hospitalized", "pregnant", "new", "continuation"])
        return has_choice_token

    def _extract_options_from_line(self, line_text: str) -> list[str]:
        options: list[str] = []

        # Checkbox style: "□ New or □ Continuation of Therapy"
        boxed = re.findall(r"□\s*([^□]+?)(?=(?:□|$))", line_text)
        for chunk in boxed:
            for piece in re.split(r"\bor\b", chunk, flags=re.IGNORECASE):
                cleaned = piece.strip(" ?:.;")
                if cleaned and len(cleaned) <= 80:
                    options.append(cleaned)

        # Yes/No style without boxes (or when checkbox not extracted).
        if re.search(r"\bYes\b", line_text, flags=re.IGNORECASE):
            options.append("Yes")
        if re.search(r"\bNo\b", line_text, flags=re.IGNORECASE):
            options.append("No")
        if re.search(r"\bOther\b", line_text, flags=re.IGNORECASE):
            options.append("Other")

        # New/Continuation style.
        if re.search(r"\bNew\b", line_text, flags=re.IGNORECASE):
            options.append("New")
        if re.search(r"\bContinuation\b", line_text, flags=re.IGNORECASE):
            options.append("Continuation of Therapy")

        # Comma/or style options (e.g., "self-administered, physician's office, or other").
        comma_or_chunks = re.split(r":", line_text, maxsplit=1)
        option_segment = comma_or_chunks[1] if len(comma_or_chunks) > 1 else line_text
        if "," in option_segment or re.search(r"\bor\b", option_segment, flags=re.IGNORECASE):
            pieces = re.split(r",|\bor\b", option_segment, flags=re.IGNORECASE)
            for piece in pieces:
                cleaned = piece.strip(" ?:.;")
                if not cleaned:
                    continue
                # Avoid capturing the left-side label text as an option.
                if len(cleaned.split()) > 7:
                    continue
                if any(token in cleaned.lower() for token in ["self-administered", "physician", "office", "other", "yes", "no", "new", "continuation"]):
                    options.append(cleaned)

        # Deduplicate while preserving order.
        deduped: list[str] = []
        seen: set[str] = set()
        for option in options:
            key = self._normalize_label(option)
            if key and key not in seen:
                seen.add(key)
                deduped.append(option)
        return deduped

    def _find_token_on_label_line(self, page: fitz.Page, token: str, label_rect: fitz.Rect) -> fitz.Rect | None:
        clip = fitz.Rect(0, max(label_rect.y0 - 8, 0), page.rect.width, min(label_rect.y1 + 8, page.rect.height))
        matches = page.search_for(token, clip=clip)
        if not matches:
            return None

        # Prefer options to the right of the label when possible.
        right_side = [rect for rect in matches if rect.x0 >= label_rect.x0 - 4]
        candidate_pool = right_side if right_side else matches

        label_center_y = (label_rect.y0 + label_rect.y1) / 2
        ranked = sorted(
            candidate_pool,
            key=lambda rect: (abs(((rect.y0 + rect.y1) / 2) - label_center_y), max(rect.x0 - label_rect.x1, 0)),
        )
        return ranked[0]
