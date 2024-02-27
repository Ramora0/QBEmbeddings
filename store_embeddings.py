from embeddings import get_embedding
import json
from tqdm import tqdm
from multiprocessing import Pool


def compute_embedding(line):
    embedding = get_embedding(line['question'])
    return line['_id'], embedding.tolist()[0]


if __name__ == "__main__":
    embeddings = {}

    with open('data/raw_questions.json', 'r') as file:
        data = json.load(file)
        data = data['tossups'][:1000]

        with Pool() as p:
            results = list(tqdm(p.imap(compute_embedding, [
                           line for line in data]), total=len(data)))

        for question, embedding in results:
            embeddings[question] = embedding

    with open('data/embeddings.json', 'w') as file:
        json.dump(embeddings, file)
