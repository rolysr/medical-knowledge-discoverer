from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from utils.anntools import Collection
from pathlib import Path


class GPT3Model:
    """GPT3 LLM for Name Entity Recognition and Relation Extraction Tasks"""
    def __init__(self) -> None:
        self.llm = OpenAI(temperature=0.7)

    def _full_task(self, sentence):
        template = """Your task is to act as a participant from eHealth-KD competition. You are given a sentencen and you must\
            \
            output the keyphrases and their classifications from the sentence. Also, extract the possible relations among those extracted entitites\
            \
            and provide the classification for that type of relation.\
            \
            For example:\
            \
            Sentence: Los glóbulos blancos ayudan a su organismo a combatir infecciones.\
            \
            Keyphrases: Concept(organismo), Action(combatir), Concept(infecciones), Concept(glóbulos blancos), Action(ayudan)\
            \
            Relations: subject(combatir, organismo), target(combatir, infecciones), subject(ayudan, glóbulos blancos), target(ayudan, combatir)\
            \
            Sentence: {sentence}\
            \
            Keyphrases:\
            \
            Relations:"""
        input_variables = {"sentence": sentence}
        return self._execute_llm(template=template, input_variables=input_variables)

    def _get_entitites_from_sentence(self, sentence):
        entities = {}
        for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                entities[text] = keyphrase.label
        return entities
    
    def _get_relations_from_sentence(self, sentence):
        relations = {}
        for relation in sentence.relations:
            origin = relation.from_phrase
            origin_text = origin.text.lower()
            destination = relation.to_phrase
            destination_text = destination.text.lower()

            relations[
                origin_text, origin.label, destination_text, destination.label
            ] = relation.label
        return relations

    def _execute_llm(self, template, input_variables):
        prompt = PromptTemplate(template=template, input_variables=list(input_variables.keys()))
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response = llm_chain.run(**input_variables)
        return response
    
    def run(self, test_dataset_path):
        collection = Collection().load_dir(Path(test_dataset_path))
        C_A, C_B, P_A, I_A, M_A, M_B, S_B = 0, 0, 0, 0, 0, 0, 0

        for sentence in collection.sentences:
            keyphrases_relations = self._full_task(sentence)
            print(keyphrases_relations)