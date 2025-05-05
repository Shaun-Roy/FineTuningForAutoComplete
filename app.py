import redis 
from flask import Flask, request, jsonify, render_template
import sqlite3
import json
import subprocess

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
app = Flask(__name__)

DB_PATH = 'autocomplete.db'

# Path to your Ollama model
ollama_model = "mistral-7b-v0.3"  # Replace with your model's name

# Set Cache TTL (time-to-live) for Redis cache
CACHE_TTL = 3600  # 1 hour

def get_cached_or_fetch(user_input, needed=5):
    # Check Redis for cached suggestions
    cached_suggestions = r.get(user_input)
    if cached_suggestions:
        print("Cache hit")
        return json.loads(cached_suggestions)

    print("Cache miss, calling Ollama")
    suggestions = get_ollama_suggestions(user_input, needed)

    # Cache the suggestions in Redis with TTL
    r.setex(user_input, CACHE_TTL, json.dumps(suggestions))

    return suggestions

def get_ollama_suggestions(user_input, needed=5):
    # Set up your system prompt here
    system_prompt = f"""
    You are an autocomplete system. Your task is to predict the next word(s) based on the user's input. 
    Given the input: '{user_input}', provide up to {needed} next word predictions in JSON format, 
    excluding the input. Example output should be: ["word1", "word2", "word3", ..., "wordN"]
    """

    # Call the Ollama model via subprocess (assuming Ollama is installed and running)
    result = subprocess.run(
        ['ollama', 'run', ollama_model, '--prompt', system_prompt],
        capture_output=True,
        text=True
    )

    # Capture the output from Ollama
    output = result.stdout.strip()

    # Parse the output into JSON format
    try:
        suggestions = json.loads(output)
        return suggestions[:needed]
    except json.JSONDecodeError as e:
        print(f"Error parsing Ollama output: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    prefix = request.args.get('prefix', '').lower()
    if not prefix:
        return jsonify({'suggestions': [], 'source': 'none'})

    # First, check Redis cache
    cached_suggestions = r.get(prefix)
    if cached_suggestions:
        print("Cache hit")
        return jsonify({'suggestions': json.loads(cached_suggestions), 'source':'redis'})

    # Query SQLite for suggestions
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT ngrams FROM ngrams WHERE ngrams LIKE ? ORDER BY count DESC LIMIT 5"
    cursor.execute(query, (prefix + '%',))
    results = cursor.fetchall()
    conn.close()

    sql_suggestions = [row[0] for row in results]

    # If there are suggestions from SQLite, cache them and return
    if sql_suggestions:
        print("SQL hit")
        r.setex(prefix, CACHE_TTL, json.dumps(sql_suggestions))
        return jsonify({'suggestions': sql_suggestions,'source':'sql'})

    # If fewer than 5 suggestions from SQLite, call Ollama
    print("SQL miss, calling Ollama")
    ollama_suggestions = get_ollama_suggestions(prefix, needed=5)

    # Cache Ollama results for future use
    r.setex(prefix, CACHE_TTL, json.dumps(ollama_suggestions))

    return jsonify({'suggestions': ollama_suggestions,'source':'llm'})

if __name__ == '__main__':
    app.run(debug=True)
