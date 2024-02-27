from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read data from files
def read_file(file_path):
    with open("C:\\Users\\deela\\Desktop\\DSGP\\SimpleAA\\WhatsappDataCleaning\\"+file_path, "r", encoding="utf-8") as file:
        return file.readlines()

# Read data for each author
janaka_sentences = read_file("Janaka IIT - Cleaned_Chat.txt")
suhasi_sentences = read_file("Suhasi IIT - Cleaned_Chat.txt")
yenuka_sentences = read_file("Yenuka IIT - Cleaned_Chat.txt")
arkash_sentences = read_file("Arkhash IIT - Cleaned_Chat.txt")
benura_sentences = read_file("Benura IIT - Cleaned_Chat.txt")
haleef_sentences = read_file("Haleef IIT - Cleaned_Chat.txt")
haroon_sentences = read_file("Haroon IIT - Cleaned_Chat.txt")
induwari_sentences = read_file("Induwari IIT - Cleaned_Chat.txt")



# Combine sentences and corresponding authors
sentences = janaka_sentences + yenuka_sentences
authors = ['Janaka'] * len(janaka_sentences) + ['Yenuka'] * len(yenuka_sentences)


# # Combine sentences and corresponding authors
# sentences = janaka_sentences[:150] + yenuka_sentences[:150] + suhasi_sentences[:150] + arkash_sentences[:150] + benura_sentences[:150] + haleef_sentences[:150] + haroon_sentences[:150] + induwari_sentences[:150] # Ensure equal number of samples

# authors = ['Janaka'] * 150 + ['Yenuka'] * 150 + ['Suhasi'] * 150 + ['Arkhash'] * 150  + ['Benura'] * 150 + ['Haleef'] * 150 + ['Haroon'] * 150 + ['Induwari'] * 150 # Equal number of samples for each author

# Transforming sentences into numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, authors, test_size=0.2, random_state=42)

# Training a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting authors for test sentences
predictions = classifier.predict(X_test)

# Function to predict author based on input sentence
def predict_author(sentence):
    vectorized_sentence = vectorizer.transform([sentence])
    author = classifier.predict(vectorized_sentence)[0]
    return author

# Test the function

test_sentence = input("Enter suspect author: ")
predicted_author = predict_author(test_sentence)
print("Predicted author text:", predicted_author)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
