from flask import Flask, request
import singlish_similarity
import english_similarity
import time
import redis
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

redis_con = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.route('/api/<lang>', methods=['POST'])
def post(lang):
    if (lang != 'en' and lang != 'si'):
        return { 'error': 'Invalid language' }, 400

    text_1 = request.json['text_1']
    text_2 = request.json['text_2']

    text_1 = text_1.replace('\n', ' ')
    text_2 = text_2.replace('\n', ' ')

    client_ip = request.remote_addr
    timestamp = str(int(time.time()))
    request_id = timestamp + '_' + client_ip

    redis_con.set(request_id, 'pending')

    if lang == 'si':
        singlish_similarity.get_similarity(request_id, text_1, text_2)
    else:
        english_similarity.get_similarity(request_id, text_1, text_2)

    return { 'id': timestamp}

@app.route('/api/<lang>/<id>', methods=['GET'])
def get(lang, id):
    if (lang != 'en' and lang != 'si'):
        return { 'error': 'Invalid language' }, 400

    client_ip = request.remote_addr
    doc_id = id + '_' + client_ip
    result = redis_con.get(doc_id)
    if result is None:
        return { 'error': 'ID not found' }, 404
    else:
        if result != 'pending': redis_con.delete(doc_id)
        return { 'result': result }