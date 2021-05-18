import pandas as pd
import streamlit as st
import numpy as np
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import requests
import json
import pickle


st.write("# Zomato Review Analyzer")
st.write("##  ")


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



def unigram(corpus, n=None):
    cv = CountVectorizer().fit(corpus)
    bag_of_words = cv.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def bigram(corpus, n=None):
    cv = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = cv.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def form1():
    with st.form(key='form1'):
        text_input = st.text_input(label="Enter your Restaurant's ID")
        submit_button = st.form_submit_button(label='Submit')
    return text_input


res_id = form1()

if res_id != "":
    header={"X-Zomato-API-Key":'7749b19667964b87a3efc739e254ada2'}
    url=f'https://developers.zomato.com/api/v2.1/reviews?res_id={res_id}&start=0&count=5'
    response=requests.get(url, headers=header)
    data=response.json()

    total_reviews = data['reviews_count']
    max = int(total_reviews/5)*5
    if max > 1000:
        max = 1000

    def form2():
        with st.form(key='form2'):
            text_input = st.text_input(label=f"Enter Prediction Dataset range (MAX: {max})")
            submit_button = st.form_submit_button(label='Submit')
        return text_input


    review_count = form2()

    if review_count != "":
        review_count = int(float(review_count))
        r = int(review_count/5)
        count = 0
        st.text('Downloading Reviews....')
        with open('test.tsv', 'w+') as f:
            f.write('Review\n')
            for i in range(r):
                url=f'https://developers.zomato.com/api/v2.1/reviews?res_id={res_id}&start={count}&count=5'
                response=requests.get(url,headers=header)
                data=response.json()
                reviews=data['user_reviews']
                for i in range(len(reviews)):
                    x = reviews[i]['review']['review_text'].strip()
                    if x != '':
                        f.write(x+'\n')

                count += 5

        st.text('Finished')


        df = pd.read_csv('test.tsv', delimiter = '\t', quoting = 3, encoding='cp1252')

        corpus=[]
        lemma = WordNetLemmatizer()
        stopwords_en = set(stopwords.words('english'))

        for review in df['Review']:
            review = review.lower()
            review = re.sub("[^a-zA-Z]", " ", review)
            review = word_tokenize(review)
            review = [word for word in review if word.lower() not in stopwords_en]
            review = [lemma.lemmatize(word) for word in review]
            review = " ".join(review)
            corpus.append(review)

        df["Review"] = corpus

        cv = CountVectorizer()
        cv = pickle.load(open('cv.pkl', 'rb'))
        bag_of_words = cv.transform(corpus).toarray()

        model = pickle.load(open('model.pkl', 'rb'))
        Label = model.predict(bag_of_words)

        Label = pd.DataFrame({'Liked':Label})
        df = df.join(Label)

        df.to_csv("test_2.tsv", sep="\t")

        st.subheader("Data Analysis/Representation")
        Labels = pd.DataFrame(df['Liked'].value_counts()).reset_index()
        Labels.columns = ['Liked','Total']
        Labels['Liked'] = Labels['Liked'].map({1: 'Positive', 0: 'Negative'})

        fig = px.pie(Labels, values = 'Total', names = 'Liked', title='Percentage of reviews', hole=.4, color = 'Liked',
                     width=800, height=400)
        st.write(fig)


        positive = df[df["Liked"] == 1][["Review", "Liked"]]
        pos_uni = unigram(positive['Review'], 20)
        temp = pd.DataFrame(pos_uni, columns = ['words' ,'count'])
        fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 unigrams in positive reviews')
        st.write(fig)


        pos_bi = bigram(positive['Review'], 20)
        temp = pd.DataFrame(pos_bi, columns = ['words' ,'count'])
        fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 bigrams in positive reviews')
        st.write(fig)


        negative = df[df["Liked"] == 0][["Review", "Liked"]]
        neg_uni = unigram(negative['Review'], 20)
        temp = pd.DataFrame(neg_uni, columns = ['words' ,'count'])
        fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 unigrams in negative reviews')
        st.write(fig)

        negative = df[df["Liked"] == 0][["Review", "Liked"]]
        neg_uni = bigram(negative['Review'], 20)
        temp = pd.DataFrame(neg_uni, columns = ['words' ,'count'])
        fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 bigrams in negative reviews')
        st.write(fig)

        with open('test_2.tsv', 'w') as f:
            f.truncate()
        with open('test.tsv', 'w') as f:
            f.truncate()
