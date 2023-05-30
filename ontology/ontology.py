import logging

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from .ontology_utils import OntologyUtils

class Ontology:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_database(self):
        relations, keyphrases = OntologyUtils.load_result()
        with self.driver.session(database="neo4j") as session:
            # Write transactions allow the driver to handle retries and transient errors
            
            for key,value in keyphrases.items():
                result = session.execute_write(
                    self.create_entity, value, key
                    )
            
            for key,value in relations.items():
                result = session.execute_write(
                    self.create_relation, key[0], key[1], key[2], key[3], value
                )
    
    @staticmethod
    def create_entity(tx, value, key):
            if value == "Concept":
                query =(
                    "CREATE (w:Concept { text: $key }) "
                    "RETURN w"
                )
            elif value == "Action":
                query =(
                    "CREATE (w:Action { text: $key }) "
                    "RETURN w"
                )
            elif value == "Predicate":
                query =(
                    "CREATE (w:Predicate { text: $key }) "
                    "RETURN w"
                )
            elif value == "Reference":
                query =(
                    "CREATE (w:Reference { text: $key }) "
                    "RETURN w"
                )
            result = tx.run(query, value=value, key=key)
    
    @staticmethod 
    def create_relation(tx, key0, key1, key2, key3, value):
            query = (
            "MATCH (p {text: $key0}), (r {text: $key2})"
            "CREATE (p)-[: is_related {relation: $value}]->(r) " 
            )
            
            result = tx.run(query, key0=key0, key1=key1, key2=key2, key3=key3, value=value)


   
    def delGraph(self):
        with self.driver.session(database="neo4j") as session:
            result = session.execute_write(
                self.delGraph_static)
    
    @staticmethod
    def delGraph_static(tx):
        query = (
            "MATCH (n)" 
            "OPTIONAL MATCH (n) - [r] - () "
            "DELETE n,r "
        )
        result = tx.run(query)
         

   

# if __name__ == "__main__":
#     # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
#     uri = "neo4j+s://8c1c0ab9.databases.neo4j.io"
#     user = "neo4j"
#     password = "F4yRbpbmrAS7ic79jwq-z-K_5gPnT7wa-LljmqnlLmA"
#     app = App(uri, user, password)
#     app.create_database()
#     app.close()