import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from flask import Flask, render_template, url_for, jsonify, request


#text_string = ''

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


app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def process():

    if request.method == 'POST':
        option == request.form['select-service']
        text = request.form['texttoinput']


        if option == 'Translate':

            result = translate(text)
            #return render_template('translate.html', translationresult = result, texttoinput = text )


        elif option == 'Summarize':
            text = request.form['text-to-input']
            result = summary(text)
            #return render_template('translate.html', textresult = result, texttoinput = text )


        elif option == 'Sentiment Analysis':
            text = request.form['text-to-input']
            blob_obj = TextBlob(text)
            sentiment_score = blob_obj.sentiment.polarity
            result = 'sentiment analyse: %.2f (-1.0 negative，1.0positive)' % sentiment_score
            #return render_template('otherservice.html', textresult=result, texttoinput=text)


        elif option == 'Keywords Extraction':
            text = request.form['text-to-input']
            keywords = analyse.extract_tags(text)
            result = 'Top3 keyword: %s' % (' / '.join(keywords[:3]))
            #return render_template('otherservice.html', textresult=result, texttoinput=text)
        return render_template('finish.html', result = result, originaltext = text)




if __name__ == '__main__':
    app.run()





