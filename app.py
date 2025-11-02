from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os
from groq import Groq
from difflib import SequenceMatcher
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Groq client from environment variable
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Create or get collection (vector database)
try:
    collection = chroma_client.get_collection("qa_collection")
except:
    collection = chroma_client.create_collection(
        name="qa_collection",
        metadata={"description": "Q&A pairs for interview practice"}
    )

# Load data into ChromaDB (run once)
def load_data():
    if collection.count() == 0:  # Only load if empty
        df = pd.read_excel('data/questions_clean.xlsx')

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            # Combine question and answer for better context
            doc = f"Question: {row['question']}\nAnswer: {row['answer']}"
            documents.append(doc)
            metadatas.append({
                "question": str(row['question']),
                "answer": str(row['answer']),
                "category": str(row['category']),
                "difficulty": str(row['difficulty'])
            })
            ids.append(f"qa_{row['id']}")

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Loaded {len(documents)} Q&A pairs into ChromaDB")

# Calculate similarity between two texts
def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get('message', '')
    mode = data.get('mode', 'answer')  # 'answer' or 'practice'

    # Search ChromaDB for relevant Q&A pairs
    results = collection.query(
        query_texts=[user_input],
        n_results=3
    )

    if mode == 'practice':
        # User is practicing - compare their answer
        correct_answer = results['metadatas'][0][0]['answer']
        similarity = calculate_similarity(user_input, correct_answer)

        feedback_prompt = f"""
        The user answered: "{user_input}"
        The correct answer is: "{correct_answer}"
        Similarity score: {similarity:.2%}

        Provide constructive feedback on the user's answer. Be encouraging but honest.
        Mention what they got right and what could be improved.
        Keep it brief (2-3 sentences).
        """

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": feedback_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7
        )

        return jsonify({
            "response": response.choices[0].message.content,
            "similarity": f"{similarity:.1%}",
            "correct_answer": correct_answer,
            "question": results['metadatas'][0][0]['question']
        })

    else:
        # User is asking a question - provide answer
        context = "\n\n".join([
            f"Q: {meta['question']}\nA: {meta['answer']}"
            for meta in results['metadatas'][0]
        ])

        prompt = f"""Based on this context:
{context}

Answer the user's question: "{user_input}"

Provide a clear, concise answer. If the question matches one from the context exactly, use that answer.
Otherwise, synthesize information from the context to help answer.
"""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.5
        )

        return jsonify({
            "response": response.choices[0].message.content,
            "sources": [meta['question'] for meta in results['metadatas'][0]]
        })

@app.route('/get_question', methods=['GET'])
def get_question():
    """Get a random question for practice"""
    all_items = collection.get()

    if len(all_items['ids']) == 0:
        return jsonify({"error": "No questions available"}), 404

    idx = random.randint(0, len(all_items['ids']) - 1)
    question = all_items['metadatas'][idx]['question']
    category = all_items['metadatas'][idx]['category']
    difficulty = all_items['metadatas'][idx]['difficulty']

    return jsonify({
        "question": question,
        "category": category,
        "difficulty": difficulty
    })

if __name__ == '__main__':
    load_data()
    app.run(debug=True, port=5000)
