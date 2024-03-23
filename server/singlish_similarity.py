import pandas as pd
from joblib import load
from feature_extration_si import extract_features
import redis
import threading

model = load('./models/singlish_similarity_model.joblib')
redis_connection = redis.Redis(host='localhost', port=6379, decode_responses=True)

def __get_similarity(id, sentence1, sentence2):
    features = extract_features(sentence1, sentence2)
    df = pd.DataFrame(features, index=[0])
    prediction = model.predict(df)[0]
    redis_connection.set(id, prediction)

def get_similarity(id, sentence1, sentence2):
    t = threading.Thread(target=__get_similarity, args=(id, sentence1, sentence2))
    t.start()