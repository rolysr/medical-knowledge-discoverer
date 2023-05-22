from py2neo import Graph
from ontology_utils import OntologyUtils


class Ontology:
    """
        Ontology class for the creation of a medical knowledge base
    """

    def __init__(self, url, password) -> None:
        """
            Class constructor. Receives the url and password parameters 
            and starts an instance capable of interacting with a Neo4J-like database
        """
        self.db = self.db = Graph(url, password=password)

    def fill_db(self, path_to_ann_file):
        """
            Method that receives a path where a file with extension .ann 
            is located and populate the database with the entities and relations in it
        """
        pass

    def make_query(self, query):
        """
            Method for sending a request to the database in
            CYPHER graph query language
        """
        if OntologyUtils.not_valid_query(query):
            raise Exception('Invalid query. Please, check the format')

        return self.db.run(query).data()
    