# File: src/neo4j_client.py

from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_GRAPH_NAME

_driver = None

def get_driver():
    """
    Singleton-style Neo4j driver.
    You usually won't use this directly after adding get_session().
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def get_session():
    """
    Always open a session on the configured database (NEO4J_GRAPH_NAME).

    Usage:
        from src.neo4j_client import get_session

        with get_session() as session:
            session.run("MATCH (n) RETURN count(n)")
    """
    driver = get_driver()
    return driver.session(database=NEO4J_GRAPH_NAME)
