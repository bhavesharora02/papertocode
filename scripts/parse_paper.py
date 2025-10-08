import sys, os, json
import fitz  # PyMuPDF
from core.ir.schema import IR

# -------- STEP 1: Extract text from PDF --------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    return text


# -------- STEP 2: Call Gemini API --------
def call_gemini_api(prompt: str) -> dict:
    import google.generativeai as genai
    import os

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    # Gemini kabhi kabhi JSON ke bahar text de deta hai
    text = response.text.strip()
    try:
        return json.loads(text)
    except Exception:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Gemini response not JSON: {text}")


# -------- STEP 3: Convert LLM output to IR --------
def llm_to_ir(text):
    prompt = f"""
You are an AI agent that extracts structured Implementation IR (Intermediate Representation) from research papers.

Input: A research paper text.
Output: A JSON object strictly following this schema (ONLY JSON, no explanation outside):

{{
  "paper": {{
    "title": "...",
    "arxiv_id": null,
    "tasks": ["..."],
    "domain": "..."
  }},
  "dataset": {{
    "name": "...",
    "source": "hf://<dataset_name>",
    "subset_fraction": 1.0,
    "splits": {{"train": 0.8, "val": 0.1, "test": 0.1}},
    "features": {{
      "text_a": "...",
      "text_b": null,
      "label": "..."
    }}
  }},
  "model": {{
    "family": "...",
    "variant": "...",
    "framework": "torch",
    "layers": [
      {{
        "type": "...",
        "params": {{"params": {{ ... }}}}
      }}
    ],
    "init": {{"pretrained": "..."}}
  }},
  "training": {{
    "loss": "...",
    "optimizer": {{"name": "...", "lr": 0.0001}},
    "scheduler": {{"name": "linear", "kwargs": {{"warmup_ratio": 0.1}}}},
    "batch_size": 32,
    "epochs": 3,
    "metrics": ["accuracy"],
    "target_metrics": {{}},
    "tolerance": 0.05
  }},
  "preprocessing": {{
    "tokenizer": "...",
    "max_len": 128
  }}
}}
"""
    return call_gemini_api(prompt + "\n\nPaper:\n" + text)


# -------- STEP 4: Save IR JSON --------
def parse_paper(pdf_path: str, outdir="artifacts/ir"):
    os.makedirs(outdir, exist_ok=True)
    raw_text = extract_text_from_pdf(pdf_path)
    ir_dict = llm_to_ir(raw_text)

    out_path = os.path.join(outdir, os.path.basename(pdf_path).replace(".pdf", "_ir.json"))
    with open(out_path, "w") as f:
        json.dump(ir_dict, f, indent=2)

    print(f"OK: Parsed {pdf_path} -> {out_path}")
    return out_path


if __name__ == "__main__":
    pdf_path = sys.argv[1]  # e.g. Samples/llama.pdf
    parse_paper(pdf_path)
