from models.gpt3_model.gpt3_model import GPT3Model

# # Basic tasks demo
# model = GPT3Model()
# k = model._extract_keyphrases("Las lágrimas son producidas por las glándulas lagrimales y el conducto lagrimal lleva estas lágrimas a los ojos.")
# r = model._extract_relations("Las lágrimas son producidas por las glándulas lagrimales y el conducto lagrimal lleva estas lágrimas a los ojos.", k)
# print(k,'\n')
# print(r)

model = GPT3Model()
model.run('./datasets/test/scenario1-main')