from openai import OpenAI
import pandas as pd
import math
import numpy as np
from siconv import sinhala_to_singlish

from io import StringIO

import pickle
import nltk

translatorPickel = open('SwaBhasha/trigramTrans.pickle', 'rb')
translator = pickle.load(translatorPickel)

# English to Sinhala ratio
def create_completion(prompt):
    API_KEY="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"
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Input likely written romanized sinhala and english.Input format: labelled as INPUT and wrapped in triple\nquotes.\nINPUT: \"\"\"input sentence\"\"\"\n\nbreak input to words,label them as sinhala,english,numeric or unknown.\n\nOutput: CSV.\nCSV order: word, is_english, is_sinhala, is_numeric_value, is_unknown\n\nfor binary,true 1,false 0.\n\nSample:\nINPUT: \"\"\"word1 word2\"\"\"\nword1, 0, 1, 0, 0\nword2, 1, 0, 0, 0\n\nExamples:\nINPUT: \"\"\"At 9.15?\"\"\"\nAt, 1, 0, 0, 0\n9.15, 0, 0, 1, 0\n\nINPUT: \"\"\"sfaewfk\"\"\"\nsfaewfk, 0, 0, 0, 1\n\nINPUT: \"\"\"biriyani ekak\"\"\"\nbiriyani, 0, 0, 0, 1\nekak, 0, 1, 0, 0\n\nINPUT: \"\"\"Beelada oi?\"\"\"\nbeelada, 0, 1, 0, 0\noi, 0, 1, 0, 0\n\nINPUT: \"\"\"Thankiuuu bn\"\"\"\nThankiuuu, 1, 0, 0, 0\nbn, 0, 1, 0, 0"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def get_word_language_df(text):
    prompt = f'INPUT: """{text}"""'
    word_language_csv = create_completion(prompt)
    word_language_csv = "word,is_english,is_sinhala,is_numeric_value,is_unknown\n" + word_language_csv
    word_language_csv = word_language_csv.replace(" ", "")
    return pd.read_csv(StringIO(word_language_csv))

def get_english_to_sinhala_ratio(df):
    try:
        return len(df[df.is_english == 1]) / len(df[df.is_sinhala == 1])
    except ZeroDivisionError:
        return 0

def triGramTranslate(sentence):
    tokenized_words = nltk.word_tokenize(sentence.lower())
    tokenized_words = [word for word in tokenized_words if word.isalpha()]
    
    translated = translator.tag(tokenized_words)
    return translated

# Singlish typing similarity
def categorize_similar_words(sentence1, sentence2):
    sentence1_translation = triGramTranslate(sentence1)
    sentence2_translation = triGramTranslate(sentence2)

    word_dictionary = {}

    for word1, trans1 in sentence1_translation:
            if trans1 in word_dictionary:
                 word_dictionary[trans1][0].append(word1)
            else:
                word_dictionary[trans1] = [[word1]]

    for word2, trans2 in sentence2_translation:
        if trans2 not in word_dictionary:
            continue # if sentence1 does not have a matching word, skip cuz we can't compare

        if len(word_dictionary[trans2]) == 1:
            word_dictionary[trans2].append([word2])
            continue

        if word2 not in word_dictionary[trans2][1]:
            word_dictionary[trans2][1].append(word2)
    
    keys_to_delete = []
    for key in word_dictionary:
        if len(word_dictionary[key]) == 1:
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        del word_dictionary[key]
    
    mapped_words = []
    for key in word_dictionary:
        mapped_words.append((word_dictionary[key][0], word_dictionary[key][1]))

    return mapped_words

def calculate_edit_distances(mapped_words):
    edit_distances = []
    for word_pairs in mapped_words:
        word1 = word_pairs[0][0]
        word2 = word_pairs[1][0]
        
        edit_distances.append(nltk.edit_distance(word1, word2))
    
    return edit_distances

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Word alignment matching
def get_aligned_word(input_word, base_word):
    mismatch_penalty = -2
    gap_penalty = -1
    match_score = 1

    input_word = " " + input_word
    base_word = " " + base_word

    m, n = len(base_word), len(input_word)

    score_matrix = np.zeros((n, m))

    for i in range(n):
        score_matrix[i][0] = i * mismatch_penalty

    for j in range(m):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, n):
        for j in range(1, m):
            max_score = max(
                score_matrix[i-1][j-1],
                score_matrix[i-1][j],
                score_matrix[i][j-1]
            )

            if input_word[i] == base_word[j]:
                score_matrix[i][j] = max_score + match_score
                continue

            # Horizontal approach, gap
            if max_score == score_matrix[i][j-1]:
                score_matrix[i][j] = max_score + gap_penalty
                continue

            # Diagonal approach, mismatch
            if max_score == score_matrix[i-1][j-1]:
                score_matrix[i][j] = max_score + mismatch_penalty
                continue
            
            # Vertical approach, mismatch
            score_matrix[i][j] = max_score + mismatch_penalty
            continue

    i, j = n-1, m-1

    aligned_input_word = ""
    aligned_base_word = ""

    while i > 0 and j > 0:
        max_score = max(
            score_matrix[i-1][j-1],
            score_matrix[i-1][j],
            score_matrix[i][j-1]
        )

        if max_score == score_matrix[i-1][j-1]:
            aligned_input_word += input_word[i]
            aligned_base_word += base_word[j]
        elif max_score == score_matrix[i][j-1]:
            aligned_input_word += "-"
            aligned_base_word += base_word[j]
        elif max_score == score_matrix[i-1][j]:
            aligned_input_word += input_word[i]
            aligned_base_word += "-"

        # update index
        if max_score == score_matrix[i-1][j-1]:
            i -= 1
            j -= 1
        elif max_score == score_matrix[i-1][j]:
            i -= 1
        else:
            j -= 1

    return (aligned_base_word[::-1], aligned_input_word[::-1]) 

def get_aligment_mismatch(text):
    translated = triGramTranslate(text)
    alignment_misses = []

    for word, translation in translated:
        if translation == 'NNN':
            continue

        aligned = list(get_aligned_word(word, sinhala_to_singlish(translation)))
        aligned[0] = aligned[0].strip()
        base_len = len(aligned[0])
        aligned[1] = aligned[1][0:base_len]

        if len(aligned[0]) == 0:
            continue
        
        try:
            aligment_disimilarity = len([i for i in aligned[1] if i == '-']) / len(aligned[0])
        except ZeroDivisionError:
            aligment_disimilarity = 1

        if aligment_disimilarity > 0.4:
            alignment_misses.append(aligment_disimilarity)
        
    try:
        return sum(alignment_misses) / len(alignment_misses)
    except ZeroDivisionError:
        return 0

def extract_features(text1, text2):
    inputs = [text1, text2]

    features = dict()

    min_input_len = min([len(i) for i in inputs])

    len_difference_threshold = 50

    for i in range(len(inputs)):
        if len(inputs[i]) > min_input_len + len_difference_threshold:
            inputs[i] = inputs[i][:min_input_len]

    # English to Sinhala ratio
    df_1 = get_word_language_df(inputs[0])
    df_2 = get_word_language_df(inputs[1])

    features['english_to_sinhala_ratio_1'] = get_english_to_sinhala_ratio(df_1)
    features['english_to_sinhala_ratio_2'] = get_english_to_sinhala_ratio(df_2)

    # Singlish typing similarity
    mapped_words = categorize_similar_words(inputs[0], inputs[1])

    edit_distances = calculate_edit_distances(mapped_words)
    avg_edit_distance = sum(edit_distances) / len(edit_distances)
    features['typing_similarity'] = 1 - sigmoid(avg_edit_distance * 0.3)

    # Word alignment matching
    features['aligment_mismatch_1'] = get_aligment_mismatch(inputs[0])
    features['aligment_mismatch_2'] = get_aligment_mismatch(inputs[1])

    return features
