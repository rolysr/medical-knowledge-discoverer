from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from utils.anntools import Collection
from pathlib import Path
import pandas as pd


class GPT3Model:
    """GPT3 LLM for Name Entity Recognition and Relation Extraction Tasks"""
    def __init__(self) -> None:
        self.llm = OpenAI(temperature=0.3)

    def _extract_keyphrases(self, sentence):
        template = """Your task is to act as a participant from eHealth-KD competition. You are given a sentencen and you must\
            \
            output the keyphrases and their classifications from the sentence.\
            \
            For example:\
            \
            Sentence: Los glóbulos blancos ayudan a su organismo a combatir infecciones.\
            \
            Keyphrases: Concept(organismo), Action(combatir), Concept(infecciones), Concept(glóbulos blancos), Action(ayudan)\
            \
            Sentence: {sentence}\
            \
            Keyphrases:"""
        input_variables = {"sentence": sentence}
        response = self._execute_llm(template=template, input_variables=input_variables)
        keyphrases = self._get_keyphrases_from_gpt3_response(response)
        return keyphrases
    
    def _get_keyphrases_from_gpt3_response(self, response):
        keyphrases = {}
        tokens = response.split('), ')
        for token in tokens:
            divided_token = token.split('(')
            keyphrase_type = divided_token[0]
            keyphrase = divided_token[1]
            if keyphrase[-1] == ')':
                split_keyphrase = keyphrase.split(')')
                keyphrase = split_keyphrase[0]
            keyphrases[keyphrase] = keyphrase_type
        return keyphrases

    def _extract_relations(self, sentence, keyphrases):
        template = """Your task is to act as a participant from eHealth-KD competition. You are given a sentencen and you must\
            \
            output the relations and their classifications from the sentence and the given keyphrases.\
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
            Keyphrases: {keyphrases}\
            \
            Relations:"""
        input_variables = {"sentence": sentence, "keyphrases": keyphrases}
        response = self._execute_llm(template=template, input_variables=input_variables)
        relations = self._get_relations_from_gpt3_response(response)
        return relations

    def _get_relations_from_gpt3_response(self, response):
        relations = {}
        tokens = response.split('), ')
        for token in tokens:
            divided_token = token.split(', ')
            keyphrase2 = divided_token[1]
            if keyphrase2[-1] == ')':
                split_keyphrase = keyphrase2.split(')')
                keyphrase2 = split_keyphrase[0]
            divided_token2 = divided_token[0].split('(')
            keyphrase1 = divided_token2[1]
            relation_type = divided_token2[0]
            relations[(keyphrase1, keyphrase2)] = relation_type
        return relations

    def _get_keyphrases_from_sentence(self, sentence):
        keyphrases = {}
        for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                keyphrases[text] = keyphrase.label
        return keyphrases
    
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
        C_A, C_B, P_A, I_A, M_A, M_B, S_A, S_B = 0, 0, 0, 0, 0, 0, 0, 0
        sentence_counter = 0

        for sentence in collection.sentences[:50]:
            sentence_counter += 1
            print("Testing sentence {}".format(sentence_counter))
            # Name Entity Recognition (Task A)
            print("Task A for sentence {}".format(sentence_counter))
            keyphrases_gold = self._get_keyphrases_from_sentence(sentence)
            keyphrases_dev = self._extract_keyphrases(sentence.text)
            c_a, p_a, i_a, m_a, s_a = self._eval_keyphrases(keyphrases_gold, keyphrases_dev)
            C_A, P_A, I_A, M_A, S_A = C_A + c_a, P_A + p_a, I_A + i_a, M_A + m_a, S_A + s_a
            
            # Relation Extraction (Task B)
            print("Task B for sentence {}".format(sentence_counter))
            relations_gold = self._get_relations_from_sentence(sentence)
            relations_dev = self._extract_relations(sentence.text, keyphrases_dev)
            c_b, m_b, s_b = self._eval_relations(relations_gold, relations_dev)
            C_B, M_B, S_B = C_B + c_b, M_B + m_b, S_B + s_b

        # Main task results
        REC_AB = (C_A + C_B + P_A/2)/(C_A + I_A + C_B + P_A + M_A + M_B)
        PREC_AB = (C_A + C_B + P_A/2)/(C_A + I_A + C_B + P_A + S_A + S_B)
        F1_AB = 2*(PREC_AB*REC_AB)/(PREC_AB + REC_AB)
        print("The results for Main Task are:\nREC_AB: {}\nPREC_AB: {}\nF1_AB: {}\n\n".format(REC_AB, PREC_AB, F1_AB))

        # Task A results
        REC_A = (C_A + P_A/2)/(C_A + I_A + P_A + M_A)
        PREC_A = (C_A + P_A/2)/(C_A + I_A + P_A + S_A)
        F1_A = 2*(PREC_A*REC_A)/(PREC_A + REC_A)
        print("The results for Task A are:\nREC_A: {}\nPREC_A: {}\nF1_A: {}\n\n".format(REC_A, PREC_A, F1_A))

        # Task B results
        REC_B = (C_B)/(C_B + M_B)
        PREC_B = (C_B)/(C_B + S_B)
        F1_B = 2*(PREC_B*REC_B)/(PREC_B + REC_B)
        print("The results for Task B are:\nREC_B: {}\nPREC_B: {}\nF1_B: {}".format(REC_B, PREC_B, F1_B))

        # Write results into a .csv
        data = {
            'REC_AB': [REC_AB],
            'PREC_AB': [PREC_AB],
            'F1_AB': [F1_AB],
            'REC_A': [REC_A],
            'PREC_A': [PREC_A],
            'F1_A': [F1_A],
            'REC_B': [REC_B],
            'PREC_B': [PREC_B],
            'F1_B': [F1_B],
        }
        df = pd.DataFrame(data)
        df.to_csv('./gpt3_results.csv')



    def _eval_keyphrases(self, gold, dev):
        c_a, p_a, i_a, m_a, s_a = 0, 0, 0, 0, 0
        keyphrases_gold = gold.keys()
        keyphrases_dev = dev.keys()
        marked_gold = {keyphrase: False for keyphrase in keyphrases_gold}

        # c_a, i_a, s_a, p_a
        for k in keyphrases_dev:
            if k in keyphrases_gold:
                if dev[k] == gold[k]:
                    c_a += 1
                else:
                    i_a += 1

            elif self._is_partial_match(k, keyphrases_gold, marked_gold):
                p_a += 1

            else:
                s_a += 1

        # m_a
        for k in keyphrases_gold:
            if k not in keyphrases_dev:
                m_a += 1
        
        return c_a, p_a, i_a, m_a, s_a

    def _is_partial_match(self, k, keyphrases_gold, marked_gold):
        for kg in keyphrases_gold:
            if (k in kg) and not marked_gold[kg]:
                marked_gold[kg] = True
                return True
        return False

    def _eval_relations(self, gold, dev):
        c_b, m_b, s_b = 0, 0, 0
        relations_dev = dev.keys()
        for relation in gold.keys():
            keyphrase1 = relation[0]
            keyphrase2 = relation[2]
            if (keyphrase1, keyphrase2) in relations_dev and dev[(keyphrase1, keyphrase2)] == gold[relation]:
                c_b += 1
            else:
                m_b += 1

        for relation in dev.keys():
            is_in_gold = False
            key1 = relation[0]
            key2 = relation[1]
            for relation2 in gold.keys():
                key1gold = relation2[0]
                key2gold = relation2[2]
                if key1 == key1gold and key2gold == key2:
                    is_in_gold = True

            if not is_in_gold:
                s_b += 1

        return c_b, m_b, s_b
