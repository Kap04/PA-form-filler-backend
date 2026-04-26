from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz


class PdfFormFiller:
    def fill_pdf(self, template_pdf_path: str | Path, field_values: dict[str, str], output_pdf_path: str | Path) -> str:
        print(f"DEBUG: fill_pdf() received field_values = {field_values}")
        document = fitz.open(str(template_pdf_path))
        try:
            has_widgets = False
            widget_labels = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                widgets = page.widgets() or []
                if widgets:
                    has_widgets = True
                for widget in widgets:
                    field_name = widget.field_name or ""
                    field_label = widget.field_label or ""
                    label = field_name if field_name else field_label
                    
                    widget_labels.append(label)
                    
                    # Try to match by field_name first, then by label
                    value = field_values.get(field_name) if field_name else None
                    if not value and field_label:
                        value = field_values.get(field_label)
                    
                    if value:
                        print(f"DEBUG: Filling widget label='{field_label}' field_name='{field_name}' with value='{value}'")
                        widget.field_value = value
                        widget.update()

            # If no widgets found, add text overlays instead
            print(f"DEBUG: has_widgets = {has_widgets}, widget_labels = {widget_labels}")
            if not has_widgets and field_values:
                print(f"DEBUG: Adding overlay text with {len(field_values)} fields")
                self._add_overlay_text(document, field_values)
            elif has_widgets and field_values:
                # Add overlay text as fallback for any unmatched fields
                print(f"DEBUG: Adding overlay text for extracted fields")
                self._add_overlay_text(document, field_values)
                self._add_overlay_text(document, field_values)

            output_path = Path(output_pdf_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            document.save(str(output_path), incremental=False, deflate=True, garbage=4)
            return str(output_path)
        finally:
            document.close()

    def _add_overlay_text(self, document: fitz.Document, field_values: dict[str, str]) -> None:
        """Add extracted field values as text overlays on the first page."""
        if document.page_count == 0 or not field_values:
            return

        page = document.load_page(0)
        y_offset = 50
        font_size = 10
        
        # Add a header for the extracted data
        page.insert_text(
            (50, y_offset),
            "EXTRACTED PRIOR AUTHORIZATION DATA:",
            fontsize=font_size + 2,
            color=(0, 0, 0),
        )
        y_offset += 25

        # Add each field value
        for field_name, field_value in field_values.items():
            if field_value:  # Only show non-empty values
                text = f"{field_name}: {field_value}"
                page.insert_text(
                    (50, y_offset),
                    text,
                    fontsize=font_size,
                    color=(0, 0, 0),
                )
                y_offset += 15

    def extract_form_schema(self, template_pdf_path: str | Path) -> dict[str, Any]:
        document = fitz.open(str(template_pdf_path))
        try:
            fields: list[dict[str, Any]] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                for widget in page.widgets() or []:
                    fields.append(
                        {
                            "name": widget.field_name or f"field_{len(fields) + 1}",
                            "label": widget.field_label or "",
                            "type": str(widget.field_type or "text"),
                            "page": page_index + 1,
                            "required": bool(widget.field_flags & 2) if widget.field_flags is not None else False,
                            "choices": list(getattr(widget, "choice_values", []) or []),
                        }
                    )
            return {"template_name": Path(template_pdf_path).stem, "page_count": document.page_count, "fields": fields}
        finally:
            document.close()
