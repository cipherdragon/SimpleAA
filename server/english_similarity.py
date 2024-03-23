import pickle
import pandas as pd

from feature_extration_en import calculate_all_features

import redis
import threading

__rf_pickle = open('./models/random_forest_model.pickle', 'rb')
__model = pickle.load(__rf_pickle)
__rf_pickle.close()

redis_connection = redis.Redis(host='localhost', port=6379, decode_responses=True)

def __compute_similarity(feature_set1, feature_set2):
    similarities = {}

    # Sentence Length
    similarities['sentence_length'] = max(0, 1 - abs(feature_set1[0] - feature_set2[0]))

    # Punctuation Frequency
    punctuation_freq1 = feature_set1[1:33]
    punctuation_freq2 = feature_set2[1:33]
    for i in range(len(punctuation_freq1)):
        similarities['punctuation_{}'.format(i+1)] = max(0, 1 - abs(punctuation_freq1[i] - punctuation_freq2[i]))

    # POS Tag Frequency
    pos_tag_freq1 = feature_set1[33:69]
    pos_tag_freq2 = feature_set2[33:69]
    for i, tag in enumerate(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']):
        similarities['pos_tag_{}'.format(tag)] = max(0, 1 - abs(pos_tag_freq1[i] - pos_tag_freq2[i]))

    # Function Word Frequency
    function_word_freq1 = feature_set1[69:196]
    function_word_freq2 = feature_set2[69:196]
    for i, word in enumerate(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']):
        similarities['function_word_{}'.format(word)] = max(0, 1 - abs(function_word_freq1[i] - function_word_freq2[i]))

    # N-gram Transition Graph Feature
    ngram_transition_feature1 = feature_set1[196:200]
    ngram_transition_feature2 = feature_set2[196:200]
    for i, feature in enumerate(['num_nodes', 'num_edges', 'avg_degree', 'density']):
        similarities['ngram_transition_{}'.format(feature)] = max(0, 1 - abs(ngram_transition_feature1[i] - ngram_transition_feature2[i]))

    # Type-Token Ratio
    similarities['type_token_ratio'] = max(0, 1 - abs(feature_set1[200] - feature_set2[200]))

    # Passive to Active Ratio
    similarities['passive_to_active_ratio'] = max(0, 1 - abs(feature_set1[201] - feature_set2[201]))

    # Gender prediction
    similarities['gender_prediction'] = 1 if feature_set1[202] == feature_set2[202] else 0

    # English variant
    #similarities['english_variant'] = 1 if feature_set1[203] == feature_set2[203] else 0

    # Double spaces after full stop
    similarities['double_spaces'] = 1 if feature_set1[203] == feature_set2[203] else 0

    return similarities

def __preprocess(similarity_dict):
    fixed_indexes = [0, 3, 10, 13, 23, 29, 32, 200, 202, 203]  # Specify the indexes you want to extract
    keys = list(similarity_dict.keys())
    values = list(similarity_dict.values())
    extracted_keys = [keys[i] for i in fixed_indexes]
    extracted_values = [values[i] for i in fixed_indexes]
    return extracted_keys, extracted_values

def __get_similarity(id, sentence1, sentence2):
    features1 = calculate_all_features(sentence1)
    features2 = calculate_all_features(sentence2)

    similarity_values = __compute_similarity(features1, features2)
    preprocessed_keys, preprocessed_values = __preprocess(similarity_values)

    df = pd.DataFrame([preprocessed_values], columns=[preprocessed_keys])

    y_pred_probability = __model['model'].predict_proba(df)
    prediction = y_pred_probability[:, 1][0]
    redis_connection.set(id, prediction)

def get_similarity(id, sentence1, sentence2):
    t = threading.Thread(target=__get_similarity, args=(id, sentence1, sentence2))
    t.start()