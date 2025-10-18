# src/extractor.py
import json
from typing import List, Dict
from llm import LLM

TRIPLE_SYSTEM_PROMPT = (
    "You are a structured information extraction model. "
    "Your task is to identify factual relationships from a given text "
    "and represent them as a list of knowledge triples in strict JSON format.\n\n"
    "Output requirements:\n"
    "1. Return ONLY valid JSON — no explanations, no Markdown formatting, no comments.\n"
    "2. Each triple must be a JSON object with exactly three string keys: "
    "'subject', 'predicate', and 'object'.\n"
    "3. Use concise, atomic phrases (no long sentences) for each field.\n"
    "4. The result must be a JSON array. If no valid triples are found, return an empty array [].\n\n"
    "Example output:\n"
    '[{"subject": "Alice", "predicate": "born_in", "object": "Paris"}]'
)


def build_triple_prompt(text: str) -> str:
    """Builds a user prompt safely without KeyError issues."""
    return f"""Read the following passage carefully and extract all factual relationships
as (subject, predicate, object) triples.

Text:
{text}

Now return ONLY a JSON array of triples following this schema:
[{{"subject": string, "predicate": string, "object": string}}]

Do not include any text, explanations, or formatting outside of the JSON.
"""


def extract_triples_from_text(llm: LLM, text: str) -> List[Dict[str, str]]:
    prompt = build_triple_prompt(text)
    raw = llm.generate_response(prompt=prompt, system=TRIPLE_SYSTEM_PROMPT, temperature=0.0, max_tokens=800)

    # Cleanup possible Markdown/code fences
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split('```', 1)[1].rsplit('```', 1)[0].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip('`').strip()

    # Parse JSON safely
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Warning: failed to parse LLM JSON output, returning empty list")
        parsed = []

    # Normalize triples
    triples = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            s = item.get('subject') or item.get('s')
            p = item.get('predicate') or item.get('p') or item.get('relation')
            o = item.get('object') or item.get('o')
            if s and p and o:
                triples.append({'subject': s.strip(), 'predicate': p.strip(), 'object': o.strip()})

    return triples
