import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

texts = ["this paper is impressive", "this approach is flawed"]
labels = torch.tensor([1, 0])

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

outputs = model(**inputs, labels=labels)
loss = outputs.loss

loss.backward()
