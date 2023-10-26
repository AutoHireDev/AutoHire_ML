import os
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)


openai.api_key = "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"

def generate_questions_answers(topic):

    prompt = f"Generate questions and answers about {topic}. Q:"


    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=3  
    )


    answers = [choice.text.strip() for choice in response.choices]


    questions = []
    for answer in answers:
        question_response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Q: {answer} A:",
            max_tokens=50,
            n=3  
        )
        question_variants = [choice.text.strip() for choice in question_response.choices]
        questions.append({"question": answer, "answer_variants": question_variants})

    return questions

@app.route('/generate', methods=['GET'])
def generate_topic_questions_answers():
    topic = request.args.get('topic')
    
    questions_answers = generate_questions_answers(topic)
    return jsonify({"topic": topic, "questions_answers": questions_answers})

if __name__ == '__main__':
    app.run(debug=True)