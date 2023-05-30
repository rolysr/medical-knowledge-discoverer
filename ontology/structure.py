from neomodel import StructuredNode, StringProperty, RelationshipTo, RelationshipFrom, config


config.DATABASE_URL = 'bolt://neo4j:test@localhost:7687'
class Entity(StructuredNode):
    
    text = StringProperty(unique_index=True)
   
    is_a = RelationshipFrom("Entity","ENTITY")
    same_as= RelationshipFrom("Entity","ENTITY")
    has_property= RelationshipFrom("Entity","ENTITY")
    part_of= RelationshipFrom("Entity","ENTITY")
    causes= RelationshipFrom("Entity","ENTITY")
    entails= RelationshipFrom("Entity","ENTITY")
    in_time = RelationshipFrom("Entity","ENTITY")
    in_place = RelationshipFrom("Entity","ENTITY")
    in_context = RelationshipFrom("Entity","ENTITY")
    subject= RelationshipFrom("Entity","ACTION")
    target = RelationshipFrom("Entity","ACTION")
    domain= RelationshipFrom("Entity","PREDICATE")
    arg= RelationshipFrom("Entity","PREDICATE")


class Concept(Entity):
   pass
  

class Action(Entity):
    pass

class Predicate(Entity):
    pass

class Reference(Entity):
    pass


