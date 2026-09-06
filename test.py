from transformers import pipeline

ner = pipeline("ner")

text = "Joko go ke Jakarta to work in Google."

result = ner(text)

print(result)
