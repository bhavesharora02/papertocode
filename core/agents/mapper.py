import json, pathlib
from typing import Dict
from core.llm.clients import make_gemini
from core.utils.cache import get_cached, set_cached
from core.utils.cleaning import clean_json_str


PROMPT_PATH = pathlib.Path("core/llm/prompts/mapper.md")

def build_prompt(concept_json: Dict) -> str:
    tmpl = PROMPT_PATH.read_text(encoding="utf-8")
    return f"""{tmpl}

CONCEPT_JSON = {json.dumps(concept_json, ensure_ascii=False)}"""

def run_mapper(concept_json: Dict, use_cache: bool = True) -> Dict:
    prompt = build_prompt(concept_json)
    if use_cache:
        cached = get_cached("A2", prompt)
        if cached is not None:
            return cached

    llm = make_gemini()
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(text, list) and text and hasattr(text[0], "content"):
        text = text[0].content
    text = text.strip().strip("`")
    text = clean_json_str(text)
    data = json.loads(text)
    set_cached("A2", prompt, data)
    return data
