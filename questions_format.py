import json

with open('data/raw_questions.json', 'r') as file:
    raw_questions = json.load(file)['tossups']

with open('data/embeddings.json', 'r') as file:
    embeddings = json.load(file)

questions = {}

for question in raw_questions:
    question_id = question['_id']
    if question_id in embeddings:
        questions[question_id] = {
            'question': question['question'],
            'answer': question['answer'],
            'embedding': embeddings[question_id]
        }
    else:
        questions[question_id] = {
            'question': question['question'],
            'answer': question['answer']
        }

with open('data/questions.json', 'w') as file:
    json.dump(questions, file)
