# from transformers import T5ForConditionalGeneration

# model = T5ForConditionalGeneration.from_pretrained("t5-base", cache_dir='./t5_base')

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-small")