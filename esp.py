import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email Spam Classifier")
input_email=st.text_area("Please enter the E-mail")
if st.button('predict'):
    #preprocess
    transformed_email=transform_text(input_email)
    #vectorize
    vector_input=tfidf.transform([transformed_email])
    #predict
    result=model.predict(vector_input)[0]
    #display
    if result==1:
        st.header('spam')
    else:
        st.header('not spam')
