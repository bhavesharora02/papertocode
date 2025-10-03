import json, pathlib
from typing import Dict
from core.llm.clients import make_gemini
from core.utils.cache import get_cached, set_cached
from core.utils.cleaning import clean_json_str

PROMPT_PATH = pathlib.Path("core/llm/prompts/interpreter.md")

def build_prompt(paper_meta: Dict, method_text: str) -> str:
    tmpl = PROMPT_PATH.read_text(encoding="utf-8")
    return f"""{tmpl}

PAPER_META = {json.dumps(paper_meta, ensure_ascii=False)}
METHOD_TEXT = <<<
{method_text}
>>>"""

def run_interpreter(paper_meta: Dict, method_text: str, use_cache: bool = True) -> Dict:
    prompt = build_prompt(paper_meta, method_text)
    if use_cache:
        cached = get_cached("A1", prompt)
        if cached is not None:
            return cached

    llm = make_gemini()
    resp = llm.invoke(prompt)
    # Expect strictly JSON
    text = resp.content if hasattr(resp, "content") else str(resp)
    # Some wrappers return list of messages; normalize
    if isinstance(text, list) and text and hasattr(text[0], "content"):
        text = text[0].content
    # Strip code fences if any
    text = text.strip().strip("`")
    print("=== RAW GEMINI RESPONSE ===")
    print(text)
    text = clean_json_str(text)
    data = json.loads(text)
    set_cached("A1", prompt, data)
    return data
