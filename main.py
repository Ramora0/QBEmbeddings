from openai import OpenAI
import json
from embeddings import embedding_distance

with open('data/questions.json', 'r') as file:
    questions = json.load(file)

questions = {k: v for k, v in questions.items() if 'embedding' in v}

p_question_id = '65ba8788ca6edbbc92a5d2a8'
prev_question = questions[p_question_id]

distances = []
for question_id, question in questions.items():
    if question_id == p_question_id:
        continue
    distance = embedding_distance(
        prev_question['embedding'], question['embedding'])
    distances.append((question_id, distance))

distances.sort(key=lambda x: x[1], reverse=True)
distances = distances[-10:]

for question_id, distance in distances:
    print(questions[question_id]['question'])
    # print(questions[question_id]['answer'])
    print(distance)
    print()

print('Prev Question')
print(prev_question['question'])
