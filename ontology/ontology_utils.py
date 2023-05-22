import csv


class OntologyUtils:
    @staticmethod
    def save_data_on_csv(data, path='data.csv'):
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    @staticmethod
    def remove_extra_spaces(sentence: str) -> str:
        return ' '.join(sentence.split())

    @staticmethod
    def not_valid_query(query):
        forbidden = ['MATCH () RETURN *']
        return any(f in query for f in forbidden)

    @staticmethod
    def split_by_space_no_quotes(sentence) -> list[str]:
        return list(filter(lambda x: x != '', sentence.split(' ')))

    @staticmethod
    def remove_quotes_if_needed(token):
        if isinstance(token, str) and token.startswith('\'') and token.endswith('\''):
            return token[1:-1]
        return token
