import nltk
import pandas as pd
import pickle
from nltk.corpus import stopwords
from textblob import TextBlob
from flask import Flask, render_template, url_for, request
from editdistance import distance
from random2 import choice
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


# SUMMARIZE

# Create a table of word frequency
def word_frequency(text_string):
    text_words = nltk.word_tokenize(text_string)
    stop_Words = set(stopwords.words('english'))
    freq_table = {}

    for word in text_words:
        if word not in stop_Words:
            if word not in freq_table.keys():
                freq_table[word] = 1
            else:
                freq_table[word] += 1
    return freq_table

# Split the text into sentences


# Calculating the sentence scores
def sent_score(text_sent, freq_table):
    Sentscore = {}
    for sent in text_sent:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freq_table.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in Sentscore.keys():
                        Sentscore[sent] = freq_table[word]
                    else:
                        Sentscore[sent] += freq_table[word]
    return Sentscore

# Calculating the average score of sentences and considering it as a threshold
def ave_score(Sentence_score):
    sum_score = 0
    for sent in Sentence_score:
        sum_score += Sentence_score[sent]
    average = int(sum_score / len(Sentence_score))
    return average

# Getting the summary
def run_summarize(sentence, Sentence_Score, threshold):
    summary = ''
    for sent in sentence:
        if sent in Sentence_Score and Sentence_Score[sent] > (threshold):
            summary += '' + sent
    return summary

# Combining all the steps and execute
def summary(text):
    freq_table = word_frequency(text)
    text_sent = nltk.sent_tokenize(text)
    score = sent_score(text_sent, freq_table)
    ave = ave_score(score)
    results = run_summarize(text_sent, score, ave)

    print(results)

# TRANSLATE
def translate(text_string):
    blob = TextBlob(text_string)
    # Detect the input language
    Ori_lan = blob.detect_language()
    # Choose language you want to translate the original language to
    tran_result = blob.translate(from_lang=Ori_lan, to='en')
    return tran_result



# SPAM MAIL RECOGNITION

def predict(text):
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    data = [text]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    return my_prediction

# SPELL CORRECTION
def find_match(source_word):
	"""Finds the best match for a source word"""

	min_dist = 100
	# min_dist = len(source_word) * 2
	optimal_words = []

	target_file = open('common_words.txt', 'r')

	# FIXME: Runtime of this is O(n^2). Can we improve this?
	for line in target_file:
		target_word = line.rstrip()

		if distance(source_word, target_word) == min_dist:
			# Add this word to the list
			optimal_words.append(target_word)

		if distance(source_word, target_word) < min_dist:
			min_dist = distance(source_word, target_word)
			# re-initialize the list, with only this word as a possible correction
			optimal_words = [target_word]

	return choice(optimal_words)




# Flask app

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def process():

    if request.method == 'POST':
        option == request.form['option']
        text = request.form['text-to-input']

        if option == 'Translate':

            result = translate(text)

        elif option == 'Summarize':
            result = summary(text)

        elif option == 'Sentiment Analysis':
            blob_obj = TextBlob(text)
            sentiment_score = blob_obj.sentiment.polarity
            result = 'sentiment analyse: %.2f (-1.0 negative，1.0positive)' % sentiment_score

        elif option == 'Keywords Extraction':
            keywords = analyse.extract_tags(text)
            result = 'Top3 keyword: %s' % (' / '.join(keywords[:3]))

        elif option == 'Spam Mail Recognition':
            result = predict(text)

        elif option == 'Spell Correction':
            for line in text:
                source_word = line.rstrip()
                result = find_match(source_word)


        return render_template('index.html', result = result, originaltext = text)
    return render_template('index.html', name = 0)




if __name__ == '__main__':
    app.run()





