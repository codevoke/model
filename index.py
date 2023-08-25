import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Загрузка предобученной модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Подготовка данных
sentences = ["Широкий просторный дом", "Большой обширный дом!"]
input_ids = tokenizer.encode(sentences, return_tensors='tf', padding=True, truncation=True)

# Генерация нового предложения
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

# Декодирование и вывод сгенерированного предложения
generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_sentence)
