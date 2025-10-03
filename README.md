# Paper2Code 🚀
Multi-Agent GenAI System that converts ML/DL research papers into runnable code.

## Week 1 Status
- ✅ Repo scaffold created
- ✅ Requirements & configs added
- ✅ Placeholder Streamlit app

## Next Steps
- Week 2: Add PDF parsing + OCR pipeline
## Week 2 — Parsing Pipeline

We added a PDF parsing module that:
- Uses `pdfplumber` to extract raw text.
- Falls back to OCR (`pytesseract`) if needed.
- Splits text into sections (abstract, method, experiments, results).
- Saves outputs into `artifacts/parsed/*.json`.

### Example
```bash
python scripts/parse_paper.py samples/sample_paper.pdf

## Week 3 — Intermediate Representation (IR)

We introduced a structured JSON-based IR that encodes:
- Paper metadata
- Dataset config
- Model architecture & layers
- Training config (loss, optimizer, scheduler, metrics)
- Preprocessing details
- Mapping to APIs

### Example IR (BERT-SST2)
See [`examples/bert_sst2.json`](examples/bert_sst2.json)

### Validation
```bash
pytest tests/test_ir.py

Week 4 — Agents (A1/A2)

A1 (Interpreter): METHOD → conceptual JSON (family, variant, layers, training, preprocessing, dataset_hints).

A2 (Mapper): conceptual JSON → concrete PyTorch/HF configs + dataset sources + API mappings.

Caching: All LLM prompts hashed; responses stored under artifacts/cache/.

Build IR from parsed paper

python -m scripts.build_ir_from_paper artifacts/parsed/bert.json
# -> artifacts/ir/bert_ir.json