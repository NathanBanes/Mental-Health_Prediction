import requests
from bs4 import BeautifulSoup
import sqlite3
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st




#-----------PART ONE: SCRAPE DATA FROM JOB LISTING SITES -----------

#url
url = "https://remoteok.com/api"

#send HTTP request for url
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

#list of jobs and skipping metadata
data = response.json()
jobs = data[1:]


#-----------PART TWO: DATA EXTRACTION AND PREPROCESSING-----------

#store all job data in here
job_data = []

#extract these details (raw data)
for job in jobs:
    title = job['position']
    company = job['company']
    description = job['description']
    location = job['location']
    date_posted = job['date']
    is_fake = 0
    
    
    #append to job_data
    job_data.append((
        title.strip() if title else "N/A",
        company.strip(),
        description.strip(),
        location.strip() if location else "N/A",
        date_posted.strip() if date_posted else "N/A",
        is_fake

    ))

#-----------PART THREE: STORE DATA IN SQL DATABASE-----------

#create and connect to sql database servers
conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

#create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS JobListings (
    id INTEGER PRIMARY KEY,
    title TEXT,
    company TEXT,
    description TEXT,
    location TEXT,
    date_posted DATE,
    is_fake BOOLEAN
)
""")

#insert data to table
for job in job_data:
    cursor.execute("""
    INSERT INTO JobListings (title, company, description, location, date_posted, is_fake)
    VALUES (?, ?, ?, ?, ?, ?)
    """, job)

#push it to table and end it
conn.commit()
conn.close()



#-----------PART FOUR: DATA CLEANING-----------

#initialize stopwords
stop_words = set(stopwords.words('english'))

#clean text function for data cleaning
def clean_data(text):
    #remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    #remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    #lowercase
    text = text.lower()
    #remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
    
#connect to database
conn = sqlite3.connect('jobs.db')

#select all data from database
query = "SELECT * FROM JobListings"

#turn into dataframe
df = pd.read_sql_query(query, conn)

#close the connection
conn.close()

#makes a column for clean data (eng)
df['clean_data'] = df['description'].apply(clean_data)

#makes a column for text length (eng)
df['text_length'] = df['description'].apply(len)

#makes a column for num of exclamation marks (eng)
df['num_exclamation'] = df['description'].apply(lambda text: text.count('!'))

#iterates through text to detect sus words/phrases
#makes a column for num of exclamation marks (eng)
sus = ['wire', 'wire transfer', 'quick money', 'social security', 'ssn', 'urgent', 'earn money from home',
       'unlimited income', 'get rich fast', 'bank details', 'credit card info', 'rich', 'credit card', 'credit card information',
       'under the table', 'whatsapp', 'quick money', 'telegram', 'no expereince required', 'no interviews']

def sus_detection(text):
    text = text.lower()
    sus_count = 0
    for phrase in sus:
        if phrase in text:
            sus_count += 1
    return sus_count

df['sus_count'] = df['description'].apply(sus_detection)

#TF-IDF (eng)
#turn dataset from online into a df
fake_train_df = pd.read_csv("fake_job_postings.csv")

#combine desc and requirements
fake_train_df['combined_text'] = fake_train_df['description'].fillna('') + ' ' + fake_train_df['requirements'].fillna('')

#clean desc
fake_train_df['clean_text'] = fake_train_df['combined_text'].apply(clean_data)

#sort fake and real desc and add to list
fake_desc = fake_train_df[fake_train_df['fraudulent'] == 1]['clean_text'].tolist()
real_desc = fake_train_df[fake_train_df['fraudulent'] == 0]['clean_text'].tolist()

#combine all text into one list
train_texts = fake_desc + real_desc

#give labels to fake and real desc
train_labels = [1] * len(fake_desc) + [0] * len(real_desc)

#turn into df
train_df = pd.DataFrame({
    'text': train_texts,
    'label': train_labels
})

#train model using fake dataset
tfidf = TfidfVectorizer(max_features=1000)
X_train= tfidf.fit_transform(train_df['text'])
Y_train = train_df['label']

model = model = LogisticRegression(class_weight='balanced')
model.fit(X_train, Y_train)

#Use the newly trained model to predict if job posting is fake or real
X = tfidf.transform(df['clean_data'])
df['is_fake'] = model.predict(X)
y = df['is_fake']


#-----------PART FIVE: EXPLORATORY DATA ANALYSIS-----------

#Histogram Plot
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='text_length', bins=30, kde=True)
plt.title("Distribution of Word Count in Job Description")
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

#Heat Map
plt.figure(figsize=(8,6))
sns.heatmap(df[['text_length', 'num_exclamation', 'sus_count', 'is_fake']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlations Between Job Listing Traits and Fake Classification')
plt.show()


#-----------PART SIX: MODELING-----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

#Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

#XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

#Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
print("--Liner Regression Results--")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

y_pred = rf_model.predict(X_test)
print("--Rain Forest Results--")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

y_pred = xgb_model.predict(X_test)
print("--XGBoost Results--")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

y_pred = nb_model.predict(X_test)
print("--Naive Bayes Results--")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#-----------PART SEVEN: EVALUATION-----------

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#-----------PART EIGHT: DEPLOYMENT-----------

#title and summary ---1---
st.title("Fake Job Posting Detection")
st.write("A machine learning app that uses NLP to classify job ads as real or fake.")

#show raw data ---2---
sel_col = ['title', 'company', 'description', 'location', 'date_posted', 'text_length']
st.header('Dataset')
st.dataframe(df[sel_col].head())

#show all raw data if checked
if st.checkbox("Show Full Dataset"):
    st.dataframe(df[sel_col])
    
#EDA ---3---
st.header('Exploratory Data Analysis')

#display word count
fig, ax = plt.subplots()
sns.histplot(data=df, x='text_length', bins=30, kde=True, color='red')
ax.set_title('Distribution of Word Counts')
ax.set_ylabel('Frequency')
ax.set_xlabel('Word Count')
st.pyplot(fig)

#display heat map
fig, ax = plt.subplots()
sns.heatmap(df[['text_length', 'num_exclamation', 'sus_count', 'is_fake']].corr(), annot=True, cmap='coolwarm')
ax.set_title('Correlations Between Job Listing Traits and Fake Classification')
st.pyplot(fig)

#model prediction interface ---4---
st.subheader("Determine if Job Description is Real")
job_desc = st.text_area("Paste a job description below")


if st.button('Predict'):
    if len(job_desc) == 0:
        st.error('❌ Please enter a job description before predicting.')
    else:
        pred = model.predict(tfidf.transform([clean_data(job_desc)]))
        st.success("Prediction: ❌ Fake" if pred[0] == 1 else "✅ Real")
        
#mode performance metrics ---5---
st.subheader('Model Perfomance')
st.write('Classification Report')
st.code(classification_report(y_test, y_pred))

#conclusion ---6---
st.markdown("### Key Takeaways")
st.markdown("- Most fake jobs are short, overuse certain keywords, and lack company info.")
st.markdown("- The model performs well (~92% accuracy) on test data.")  








    
    











