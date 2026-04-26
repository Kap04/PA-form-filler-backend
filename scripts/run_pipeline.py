from __future__ import annotations

import argparse
from pathlib import Path

from app.services.pipeline import PAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PA extraction and form-fill pipeline")
    parser.add_argument("--emr", required=True, help="Path to the patient EMR PDF")
    parser.add_argument("--form", required=True, help="Path to the PA form PDF")
    args = parser.parse_args()

    pipeline = PAPipeline()
    try:
        result = pipeline.process(Path(args.emr), Path(args.form))
        print(f"job_id={result.job_id}")
        print(f"template_name={result.template_name}")
        print(f"output_pdf={result.output_pdf_path}")
        print(f"tracker_entry={result.tracker_entry}")
        for field in result.extracted_fields:
            print(f"field={field.name} value={field.value} confidence={field.confidence}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
