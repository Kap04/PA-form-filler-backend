# PA SaaS Backend

This backend powers a prior authorization assistant that:

1. Extracts text from a patient EMR PDF.
2. Detects the active PA form template and its writable fields.
3. Uses Mistral to map EMR evidence into the PA form schema.
4. Writes values back into the PDF while preserving form fields.
5. Stores a lightweight job tracker entry for downstream UI display.

## Run

1. Create a virtual environment.
2. Copy `.env.example` to `.env` and set `MISTRAL_API_KEY`.
3. Install dependencies with `pip install -e .`.
4. Start the API with `uvicorn app.main:app --reload --port 8000`.

## CLI pipeline

Use `python scripts/run_pipeline.py --emr path/to/emr.pdf --form path/to/form.pdf` to run the pipeline without the API.
