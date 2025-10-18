from typing import List, Dict


import json
from llm import LLM

def llm_build_context_from_triples(llm: LLM, triples: List[Dict[str, str]], max_sentences: int = 20) -> str:
    """
    Use the LLM to convert structured triples into a concise, factual text summary that captures all relationships comprehensively.
    """
    if not triples:
        return ""

    triple_text = json.dumps(triples, indent=2, ensure_ascii=False)

    system_prompt = (
        "You are an expert knowledge summarizer tasked with creating accurate, comprehensive, and natural summaries from structured data. "
        "Your input is a JSON list of triples, each containing a subject, predicate, and object, representing factual relationships. "
        "Your task is to generate a concise, coherent, and factual summary that captures *all* relationships between entities in a natural and logical flow. "
        "Follow these rules:\n"
        "1. Include *every* fact from the triples without omitting any relationships or details, even if subjects vary slightly (e.g., 'Albert Einstein' vs. 'Einstein' refers to the same entity).\n"
        "2. Use clear, declarative sentences that precisely reflect the predicates (e.g., 'is' for profession, 'born in' for birthplace, 'developed' for contributions).\n"
        "3. Group related triples to form a cohesive narrative, avoiding redundancy and ensuring logical connections between facts.\n"
        f"4. Limit the summary to approximately {max_sentences} sentences, balancing completeness and conciseness.\n"
        "5. Do not infer, fabricate, or claim ignorance of any facts explicitly present in the triples (e.g., do not say 'I don’t know' if the information is provided).\n"
        "6. Ensure the summary reads naturally as a well-structured paragraph, not a mechanical list of facts.\n"
        "7. Cross-reference all triples to ensure no information is overlooked, prioritizing commonly queried attributes like birthplace, profession, contributions, and achievements.\n"
        "8. Treat variations in subject names (e.g., 'Albert Einstein' and 'Einstein') as referring to the same entity unless explicitly contradicted.\n"
        "9. If the context suggests a query (e.g., about birthplace or achievements), prioritize and explicitly address those attributes using the triples."
    )

    user_prompt = (
        f"Input triples:\n{triple_text}\n\n"
        "Generate a coherent summary paragraph that comprehensively describes *all* relationships between entities, ensuring no facts are omitted. "
        "Use natural language and logical flow, explicitly including key attributes like birthplace, profession, contributions, and achievements. "
        "Treat 'Albert Einstein' and 'Einstein' as the same entity unless otherwise specified."
    )

    try:
        summary = llm.generate_response(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.05,  # Further lowered for maximum fidelity
            max_tokens=600     # Increased slightly to ensure all facts are included
        )
        return summary.strip()
    except Exception as e:
        print(f"[WARN] Failed to generate context from triples: {e}")
        return ""