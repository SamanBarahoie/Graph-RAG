# src/graph_rag.py
import json
from typing import List
from llm import LLM
from neo4j_client import Neo4jClient
from context import llm_build_context_from_triples

ENTITY_EXTRACTION_SYSTEM = (
    "You are a structured entity extraction model. "
    "Your task is to identify the key named entities and domain-specific concepts mentioned in a user question. "
    "\n\n"
    "Output requirements:\n"
    "1. Return ONLY valid JSON — no text, no explanations, no Markdown formatting.\n"
    "2. Output must be a JSON array of strings, each representing one entity or concept.\n"
    "3. Use concise canonical forms (e.g., 'Albert Einstein', not 'the famous physicist Albert Einstein').\n"
    "4. If no entities are found, return an empty array [].\n\n"
    "Example output:\n"
    "[\"Albert Einstein\", \"Theory of Relativity\"]"
)


ENTITY_EXTRACTION_PROMPT = (
    "Identify all main entities and key domain concepts mentioned in the following question.\n\n"
    "Question:\n{question}\n\n"
    "Return ONLY a JSON array of entity strings.\n"
    "Example: [\"Quantum Mechanics\", \"Niels Bohr\"]\n\n"
    "Do not include any text or explanation outside the JSON output."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a factual reasoning assistant that answers user questions using the provided context.\n\n"
    "Guidelines:\n"
    "1. Base your answer ONLY on the given context. Do not fabricate or assume facts.\n"
    "2. If the context does not provide enough information to answer the question, say: "
    "'I don’t have enough information to answer that.'\n"
    "3. Provide clear, concise, and factual answers.\n"
    "4. If possible, suggest one follow-up question or next step.\n\n"
    "Example style:\n"
    "Answer: Albert Einstein developed the theory of relativity.\n"
    "Follow-up: You could explore how Einstein's theories impacted modern physics."
)

def answer_question_graph_rag(llm: LLM, neo4j_client: Neo4jClient, question: str) -> str:
    """
    Uses RAG over a Neo4j knowledge graph to answer a user question.
    """
    # === Step 1: Entity extraction ===
    raw_entities = llm.generate_response(
        prompt=ENTITY_EXTRACTION_PROMPT.format(question=question),
        system=ENTITY_EXTRACTION_SYSTEM,
        temperature=0.0,
        max_tokens=200
    )

    try:
        entities = json.loads(raw_entities)
        if not isinstance(entities, list):
            entities = []
    except Exception:
        entities = [e.strip() for e in raw_entities.replace('\n', ',').split(',') if e.strip()]

    if not entities:
        return "I couldn’t identify any entities to search for. Please rephrase your question."

    # === Canonicalize entities ===
    canonical_entities = []
    for ent in entities:
        canon = neo4j_client.get_canonical_entity(ent)
        if canon and canon not in canonical_entities:
            canonical_entities.append(canon)
        elif ent not in canonical_entities:
            canonical_entities.append(ent)

    # === Step 2: Collect triples from Neo4j ===
    collected_triples = []
    for entity in canonical_entities:
        try:
            rows = neo4j_client.fetch_neighborhood(entity, depth=2, limit=10)
            for r in rows:
                if "triples" in r:
                    collected_triples.extend(r["triples"])
        except Exception as e:
            print(f"[WARN] Neo4j fetch_neighborhood failed for {entity}: {e}")
            continue

    # Try connecting entities pairwise
    if len(canonical_entities) >= 2:
        for i in range(len(canonical_entities) - 1):
            a, b = canonical_entities[i], canonical_entities[i + 1]
            try:
                paths = neo4j_client.find_path_between(a, b, max_hops=3)
                for p in paths:
                    if "triples" in p:
                        collected_triples.extend(p["triples"])
            except Exception as e:
                print(f"[WARN] Neo4j find_path_between failed for {a}, {b}: {e}")
                continue

    # === Step 3: Deduplicate triples ===
    seen = set()
    unique_triples = []
    for t in collected_triples:
        key = (t.get("subject"), t.get("predicate"), t.get("object"))
        if None in key or key in seen:
            continue
        seen.add(key)
        unique_triples.append(t)

    # === Step 4: Build context with LLM ===
    context = llm_build_context_from_triples(llm, unique_triples, max_sentences=8)

    # === Step 5: Ask final question ===
    final_prompt = (
        f"Context:\n{context if context else '(no relevant facts found)'}\n\n"
        f"Question:\n{question}\n\n"
        "Answer concisely based only on the context. "
        "If the context lacks relevant facts, say you don’t know."
    )

    answer = llm.generate_response(
        prompt=final_prompt,
        system=ANSWER_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=400
    )

    return answer.strip()
