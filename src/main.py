from llm import LLM
from config import CONFIG
from neo4j_client import Neo4jClient
from extractor import extract_triples_from_text
from graph_rag import answer_question_graph_rag

if __name__ == '__main__':
    cfg = CONFIG

    if not cfg.get('API_KEY'):
        raise RuntimeError('API_KEY not set in config.py')

    # Initialize LLM
    llm = LLM(model=cfg['MODEL'], api_key=cfg['API_KEY'])

    # Initialize Neo4j client
    neo = Neo4jClient(uri=cfg['NEO4J_URI'], user=cfg['NEO4J_USER'], password=cfg['NEO4J_PASSWORD'])

    # Sample document ingestion
    doc = (
        "Albert Einstein, a renowned theoretical physicist born in Ulm, developed the theory of relativity and worked on the photoelectric effect, which earned him the Nobel Prize. "

    )

    triples = extract_triples_from_text(llm, doc)
    print('Extracted triples:', triples)

    for t in triples:
        neo.upsert_triple(t['subject'], t['predicate'], t['object'])

    # Sample Graph-RAG question
    q = 'What is Albert Einstein known for and where was he born?'
    ans = answer_question_graph_rag(llm, neo, q)
    print('\nGraph-RAG Answer:\n', ans)

    neo.close()
