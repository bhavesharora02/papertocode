import json
import google.generativeai as genai
from core.utils.cleaning import clean_json_str

def run_debugger(ir_data, report, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key, transport="rest")

    # Use supported model IDs only
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
    except Exception:
        model = genai.GenerativeModel("gemini-2.5-flash")


    prompt = f"""
    You are a Debugging Agent.

    Original IR:
    {json.dumps(ir_data, indent=2)}

    Verification Report:
    {json.dumps(report, indent=2)}

    Paper Target Metrics:
    {json.dumps(ir_data.get("training", {}).get("target_metrics", {}), indent=2)}

    Suggest JSON patch to improve performance.
    Rules:
    - Strict JSON only (no markdown, no comments).
    - Allowed fixes: adjust epochs, learning rate, batch size, add simple layers.
    """

    resp = model.generate_content(prompt)
    text = clean_json_str(resp.text)
    return json.loads(text)

def apply_patch(ir_data, patch):
    """Merge patch into existing IR data."""
    if "training" in patch:
        ir_data["training"].update(patch["training"])
    if "suggested_layers" in patch and not ir_data["model"].get("layers"):
        ir_data["model"]["layers"] = patch["suggested_layers"]
    return ir_data
