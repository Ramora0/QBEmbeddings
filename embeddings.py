from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs[0].mean(dim=1).numpy()


def embedding_distance(embedding1, embedding2):
    return cosine(embedding1, embedding2)
