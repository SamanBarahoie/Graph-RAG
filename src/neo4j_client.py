# neo4j_client.py
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any, Optional

class Neo4jClient:
    """
    Improved Neo4j client with:
    - upsert_triple
    - neighborhood fetch with candidate resolution
    - shortest-path search using safe parameter names
    - canonical entity lookup and contains-based candidate search
    """

    def __init__(self, uri: str, user: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            # smoke test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("[INFO] Connected to Neo4j successfully.")
        except Exception as e:
            print(f"[ERROR] Could not connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("[INFO] Neo4j connection closed.")

    # ---------- Canonical / candidate lookup ----------
    def _get_exact_entity(self, name: str) -> Optional[str]:
        if not self.driver:
            return None
        query = "MATCH (e:Entity {name: $name}) RETURN e.name AS name LIMIT 1"
        with self.driver.session() as session:
            rec = session.run(query, name=name).single()
            return rec["name"] if rec else None

    def find_entity_candidates(self, name: str, limit: int = 5) -> List[str]:
        if not self.driver:
            return []
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e.name AS name LIMIT $limit
        """
        with self.driver.session() as session:
            return [r["name"] for r in session.run(query, name=name, limit=limit)]

    def get_canonical_entity(self, name: str) -> Optional[str]:
        """
        Resolve token to canonical name with strategy:
          1) exact match
          2) case-insensitive exact match
          3) first contains candidate
        """
        if not self.driver:
            return None

        exact = self._get_exact_entity(name)
        if exact:
            return exact

        # case-insensitive exact
        query_ci = "MATCH (e:Entity) WHERE toLower(e.name) = toLower($name) RETURN e.name AS name LIMIT 1"
        with self.driver.session() as session:
            rec = session.run(query_ci, name=name).single()
            if rec:
                return rec["name"]

        # contains
        candidates = self.find_entity_candidates(name, limit=1)
        return candidates[0] if candidates else None

    # ---------- Upsert triple ----------
    def upsert_triple(self, subject: str, predicate: str, object_: str) -> None:
        if not self.driver:
            raise ConnectionError("Neo4j driver is not initialized.")
        query = """
        MERGE (a:Entity {name: $subj})
        MERGE (b:Entity {name: $obj})
        MERGE (a)-[r:REL {type: $rel}]->(b)
        ON CREATE SET r.created = timestamp()
        """
        try:
            with self.driver.session() as session:
                session.run(query, subj=subject, obj=object_, rel=predicate)
        except Exception as e:
            print(f"[ERROR] Failed to upsert triple ({subject}, {predicate}, {object_}): {e}")

    # ---------- Fetch neighborhood (with candidate resolution + dedupe) ----------
    def fetch_neighborhood(self, entity: str, depth: int = 2, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Resolve entity to canonical/candidates and fetch neighbourhood for each candidate.
        Returns list of {'triples': [...]}. Triples deduped across candidates.
        """
        if not self.driver:
            raise ConnectionError("Neo4j driver is not initialized.")

        canonical = self.get_canonical_entity(entity)
        candidates = [canonical] if canonical else self.find_entity_candidates(entity, limit=5)

        seen = set()
        rows: List[Dict[str, Any]] = []

        for cand in candidates:
            if not cand:
                continue
            query = f"""
            MATCH p=(e:Entity {{name:$name}})-[*1..{depth}]-(x)
            RETURN p LIMIT $limit
            """
            try:
                with self.driver.session() as session:
                    result = session.run(query, name=cand, limit=limit)
                    for record in result:
                        path = record["p"]
                        triples = []
                        for i in range(len(path.nodes) - 1):
                            a = path.nodes[i]
                            b = path.nodes[i + 1]
                            rel = path.relationships[i]
                            subj = a.get("name") if hasattr(a, "get") else a["name"]
                            # relationship type: prefer rel.type (Neo4j relationship type)
                            rel_type = None
                            try:
                                # if relationship has a 'type' property, prefer it
                                rel_type = rel.get("type") if hasattr(rel, "get") and rel.get("type") else getattr(rel, "type", None)
                            except Exception:
                                rel_type = getattr(rel, "type", None)
                            obj = b.get("name") if hasattr(b, "get") else b["name"]
                            key = (subj, rel_type, obj)
                            if key in seen:
                                continue
                            seen.add(key)
                            triples.append({"subject": subj, "predicate": rel_type, "object": obj})
                        if triples:
                            rows.append({"triples": triples})
            except Exception as e:
                print(f"[WARN] fetch_neighborhood for {cand} failed: {e}")
                continue

        return rows

    # ---------- Find shortest path between two entities ----------
    def find_path_between(self, from_entity: str, to_entity: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """
        Resolve canonical names for both endpoints and search shortestPath.
        Uses safe parameter names ($from_name / $to_name).
        """
        if not self.driver:
            raise ConnectionError("Neo4j driver is not initialized.")

        a_name = self.get_canonical_entity(from_entity) or from_entity
        b_name = self.get_canonical_entity(to_entity) or to_entity

        query = f"""
        MATCH p = shortestPath( (a:Entity {{name:$from_name}})-[*..{max_hops}]-(b:Entity {{name:$to_name}}) )
        RETURN p
        """

        paths: List[Dict[str, Any]] = []
        try:
            with self.driver.session() as session:
                # pass parameters via dict to avoid python keyword collisions
                result = session.run(query, {"from_name": a_name, "to_name": b_name})
                for record in result:
                    path = record["p"]
                    triples = []
                    for i in range(len(path.nodes) - 1):
                        a = path.nodes[i]
                        b = path.nodes[i + 1]
                        rel = path.relationships[i]
                        subj = a.get("name") if hasattr(a, "get") else a["name"]
                        try:
                            rel_type = rel.get("type") if hasattr(rel, "get") and rel.get("type") else getattr(rel, "type", None)
                        except Exception:
                            rel_type = getattr(rel, "type", None)
                        obj = b.get("name") if hasattr(b, "get") else b["name"]
                        triples.append({"subject": subj, "predicate": rel_type, "object": obj})
                    paths.append({"triples": triples})
        except Exception as e:
            print(f"[WARN] find_path_between failed for {a_name} <-> {b_name}: {e}")

        return paths
