import re

def clean_json_str(s: str) -> str:
    """
    Clean Gemini output to make sure it's valid JSON.
    - Remove markdown-style fences
    - Remove leading 'json' keyword
    """
    s = s.strip()
    # Remove leading/trailing fences
    s = re.sub(r"^```(json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s, flags=re.IGNORECASE).strip()
    # Remove a lone leading 'json'
    if s.lower().startswith("json"):
        s = s[4:].strip()
    return s
