import numpy as np
import nltk
from collections import Counter
from nltk.util import bigrams
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag
import networkx as nx
import string
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import openai
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

lstm_tokenizer_pickle = open("./models/lstm_tokenizer.pickle", "rb")
tokenizer_model = pickle.load(lstm_tokenizer_pickle)
lstm_tokenizer_pickle.close()

label_encoder_pickle = open("./models/lstm_label_encoder.pickle", "rb")
label_encoder_model = pickle.load(label_encoder_pickle)
label_encoder_pickle.close()

max_length_pickle = open("./models/max_length.pickle", "rb")
max_length_model = pickle.load(max_length_pickle)
max_length_pickle.close()

gender_model = load_model("./models/lstm_trained_model.h5")

def __sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:  # Handle case where there are no sentences in the text
        return 0.0  # Return 0 if there are no sentences
    lengths = [len(sentence.split()) for sentence in sentences]
    min_length = min(lengths)
    max_length = max(lengths)
    if min_length == max_length:  # Handle case where all sentences have the same length
        return 0.0  # Return 0 if all sentences have the same length
    avg_sentence_length = sum(lengths) / len(sentences)
    normalized_length = (avg_sentence_length - min_length) / (max_length - min_length)
    return normalized_length

def __calculate_punctuation_frequency(text):
    punctuation_marks = set(string.punctuation)
    punctuation_counts = Counter(char for char in text if char in punctuation_marks)
    total_punctuation = sum(punctuation_counts.values())
    punctuation_distribution = {punct: count / total_punctuation for punct, count in punctuation_counts.items()}
    return [punctuation_distribution.get(mark, 0) for mark in punctuation_marks]

def __calculate_pos_tag_frequency(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tag_counts = Counter(tag for word, tag in pos_tags)
    total_pos_tags = sum(pos_tag_counts.values())
    pos_tag_distribution = {tag: count / total_pos_tags for tag, count in pos_tag_counts.items()}
    all_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    return [pos_tag_distribution.get(tag, 0) for tag in all_tags]

def __calculate_function_word_frequency(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    function_words_text = [word for word in tokens if word.lower() in stop_words]
    total_function_words = len(function_words_text)
    function_word_counts = Counter(function_words_text)
    function_word_frequencies = {word: count / total_function_words for word, count in function_word_counts.items()}
    all_function_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    return [function_word_frequencies.get(word, 0) for word in all_function_words]

def __is_passive_voice(tagged_sentence):
    for i in range(1, len(tagged_sentence)):
        if (
            tagged_sentence[i][0] == "by" and
            tagged_sentence[i - 1][1].startswith("V") and
            tagged_sentence[i][1] == "IN"
        ):
            return True
    return False

def __calculate_passive_to_active_ratio(text):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    passive_count = 0
    active_count = 0

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_sentence = nltk.pos_tag(words)

        if __is_passive_voice(tagged_sentence):
            passive_count += 1
        else:
            active_count += 1

    return passive_count / active_count if active_count > 0 else 0

def __is_passive_to_binary(passive_to_active_ratio):
    return 1 if passive_to_active_ratio > 1 else 0

def __ngram_transition_graph_feature(text, n=5):
    tokens = nltk.word_tokenize(text)
    ngrams = list(nltk.ngrams(tokens, n))
    transition_graph = nx.DiGraph()
    transition_graph.add_nodes_from(ngrams)
    for i in range(len(ngrams) - 1):
        transition_graph.add_edge(ngrams[i], ngrams[i + 1])

    # Compute graph properties
    num_nodes = transition_graph.number_of_nodes()
    num_edges = transition_graph.number_of_edges()
    avg_degree = np.mean([val for (node, val) in transition_graph.degree()])
    density = nx.density(transition_graph)

    # Return computed graph properties as a feature vector
    return np.array([num_nodes, num_edges, avg_degree, density])

def __type_token_ratio(text):
    tokens = nltk.word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)

def __predict_gender(text):
    new_data_sequence = tokenizer_model.texts_to_sequences([text])
    new_data_padded = pad_sequences(new_data_sequence, maxlen=max_length_model)

    prediction = gender_model.predict(new_data_padded)
    predicted_class = (prediction > 0.5).astype('int')[0][0]

    return predicted_class

def __detect_english_variant(text):
    prompt = "Please analyze the language and phrasing of the paragraph provided below and determine whether it aligns more closely with American English or British English.\n\n" + text + "\n\nLanguage variant:"

    # Set up OpenAI API
    openai.api_key = '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'

    # Use GPT-3.5 to determine English variant
    response = openai.Completion.create(
      engine="gpt-3.5-turbo-instruct",
      prompt=prompt,
      temperature=0,
      max_tokens=800
    )

    # Extracting the prediction from the response
    prediction_text = response.choices[0].text.strip()
    print("Prediction text: ", prediction_text)

    # Assign numeric values to the outcomes
    if prediction_text == "American English":
        return 1
    elif prediction_text == "British English":
        return 2
    else:
        return 0  # Return 0 for other cases or errors

def __check_double_spaces_after_full_stop(text):
    double_spaces_count = text.count(".  ")
    if double_spaces_count >= 3:
        return 1
    else:
        return 0


def calculate_all_features(text):
    features = []

    features.append(__sentence_length(text))
    features.extend(__calculate_punctuation_frequency(text))
    features.extend(__calculate_pos_tag_frequency(text))
    features.extend(__calculate_function_word_frequency(text))
    features.extend(__ngram_transition_graph_feature(text))
    features.append(__type_token_ratio(text))

    passive_to_active_ratio = __calculate_passive_to_active_ratio(text)
    features.append(__is_passive_to_binary(passive_to_active_ratio))

    features.append(__predict_gender(text))

    double_spaces = __check_double_spaces_after_full_stop(text)
    features.append(double_spaces)

    return features

