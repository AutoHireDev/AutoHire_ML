import openai
import json,os

from flask import Flask, request, jsonify
from flask_cors import CORS

openai.api_key = os.getenv("OPENAI_KEY")

app = Flask(__name__)
CORS(app)

def generate_questions_and_answers(topic):
    prompt = f"Generate 2 interview questions and 3 possible answers for each question about the {topic} and return these questions and answers in JSON format only as an array of python objects with 'question' and 'answer_variants' as properties. Enclose property names in double quotes. Reply in JSON form and include no other commentary strictly. Do not include double quoutes in answers."

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000 
    )
    
    generated_text = response.choices[0].text
    return json.loads(generated_text)

@app.route('/generate', methods=['GET'])
def generate_topic_questions_answers():
    topic = request.args.get('topic')

    questions_and_answers = generate_questions_and_answers(topic)
    return questions_and_answers

if __name__ == '__main__':
    app.run(debug=True, port=5008)
