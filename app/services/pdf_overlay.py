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
                
                print(f"DEBUG: extract_form_labels() found {len(labels)} labels via AI = {labels[:10]}...")
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

                rect = label_positions.get(self._normalize_label(label))
                if not rect:
                    print(f"DEBUG: No position found for label '{label}'")
                    continue

                page_index, rect_obj = rect
                page = document.load_page(page_index)

                # Get AI-determined label type
                label_norm = self._normalize_label(label)
                label_type = self.label_metadata.get(label_norm, {}).get("type", "text")
                label_options = self.label_metadata.get(label_norm, {}).get("options", [])

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
        positions: dict[str, tuple[int, fitz.Rect]] = {}
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_text = page.get_text("text") or ""
            for label in labels_to_find:
                normalized_label = self._normalize_label(label)
                if not normalized_label or normalized_label in positions:
                    continue

                label_variants = self._label_variants(label)
                found_rect = None
                for variant in label_variants:
                    search_results = page.search_for(variant)
                    if search_results:
                        found_rect = self._choose_best_label_rect(page, label, search_results)
                        break

                if found_rect is None:
                    # Fallback: look for the label text in the extracted page text lines.
                    for line in page_text.splitlines():
                        line_normalized = self._normalize_label(line)
                        label_norm = self._normalize_label(label)
                        # Only use substring matching for longer text or exact matches to avoid collisions
                        # e.g., "direction" should not match "directions for use"
                        if line_normalized == label_norm or (
                            len(label_norm) > 20 and label_norm in line_normalized
                        ):
                            search_results = page.search_for(line.strip())
                            if search_results:
                                found_rect = self._choose_best_label_rect(page, label, search_results)
                            break

                if found_rect is not None:
                    positions[normalized_label] = (page_index, found_rect)
                    print(f"DEBUG: Found label '{label}' at {found_rect}")

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
