import pandas as pd
import re
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st


#FILTERS and CLEANS DATA
#load data
df = pd.read_csv('data_to_be_cleansed.csv')

#filter out posts with no text
df = df.dropna(subset=['title', 'text'])

#combine title and text column
df['combination'] = df['title'] + ' ' + df['text']

#clean/organize text
def clean_text(text):
    '''
    clean text to make it more readable. removes URLs and removes any sepcial 
    characters.
    
    Methods:
    .sub (extracts only given characters)
    .lower (lowercase)
    '''
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'[^a-zA-z\s]', '', text)
    return text.lower()

#makes clean_text column and applys this func to combination
df['clean_text'] = df['combination'].apply(clean_text)

#ANALYZE SENTIMENTAL VALUE OF DATA
#analyze sentimal values of text
def analyze_data(text):
    '''
    Finds negative, neutral, positive, and compound values of the text. Then we
    evaluate based on the compound score if their emotions are more on the pos,
    neg, or neu side.
    
    Methods:
    .sentiment (analyzes sentiment)
    .polarity (gives score between -1.0 - 1.0)
    '''
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.05:
        return 'positive'
    elif polarity <= -0.05:
        return 'negative'
    else:
        return 'neutral'
    
df['polarity_score'] = df['clean_text'].apply(lambda text: TextBlob(text).sentiment.polarity)

#makes sentiment column and applys this func to clean text
df['sentiment'] = df['clean_text'].apply(analyze_data)
    
#make data into plot
sns.countplot(data = df, x = 'sentiment')
plt.title('Sentiment Distribution of Reddit Mental Health Posts')
plt.show()
    
#DISPLAY DATA USING STREAMLIT
#display title and problem statement
st.title("Reddit Mental Health Prediction")
st.header("Problem Statement")
st.write("Mental health is leading cause to several issues that people face; Signs can be apperent online. This project aims to automatically analyze the sentiment of a mental health-related Reddit post using natural language processing. The goal is to help researchers, moderators, or users understand and identify emotional trends in online communities.")

#show scatter plot of polarity scores
st.header("Dataset")
st.subheader("Polarity Distribution")
st.write('Project analyzes polarity of reddit post between -1 to 1 based on differnt aspects such as phrasing.')
st.scatter_chart(df['polarity_score'])

#show sentiment counts
st.subheader('Sentiment Bar Graph')
st.write('Polarity value is converted into a sentiment based on numeric comparison.')
sentiments_counts = df['sentiment'].value_counts()

#creates bar graph plt to add color to each sentiment
fig, ax = plt.subplots()
colors = ['green', 'red', 'blue']  # Adjust depending on your sentiments
ax.bar(sentiments_counts.index, sentiments_counts.values, color=colors)
ax.set_title('Sentiment Bar Graph')
ax.set_ylabel('Count')
ax.set_xlabel('Sentiment')
st.pyplot(fig)

#methodology section
st.header("Methodology")
st.subheader("1. Features")
st.markdown("""
Used a **Natural Language Processing** pipeline in order to:

- Clean Data by removing posts with null values, removing URLs and special characters
- Run Sentiment Analysis using **TextBlob** to obtain polarity scores to then convert to sentiments

""")

st.subheader("2. Algorithms")
st.markdown("""

- **Textblobs** uses a **Naive Bayes classifier**
- Calculates Polarity (sentiment score from -1 to 1)

""")

st.subheader("3. Metrics")
st.markdown("""
##### Polarity Score:

- Numerical value from -1 to 1
- Assist in determining sentiment classificiation

##### Sentiment Count

- Number of posts classified in positive, negative, and nuetral
- Assist in determining sentiment classificiation

##### Visual Analysis

- Scatter plot of polarity scores to showcase distribution and variance across posts

""")

#results section
st.header("Results")
st.markdown("""
After analyzing Reddit posts related to mental health, the following results were observed:

- A large portion of the posts were **positive**, showing encouragement, support, or personal growth.
- A significant number of **negative** posts were also present, reflecting emotional struggles or distress.
- A smaller but noticeable number of posts were **neutral**, often containing factual or informational content without strong emotional tone.

These results are visualized in the pie chart below, which shows the proportion of each sentiment across all analyzed posts.
""")

sentiments_df = df['sentiment'].value_counts().reset_index()
sentiments_df.columns = ['Sentiment', 'Count']

fig = px.pie(sentiments_df, names='Sentiment', values='Count', title='Sentiment Distribution')
st.plotly_chart(fig)

#limitation and improvement section
st.header("Limitations and Improvements")
st.subheader("Limitations")
st.markdown("""
            
- **Textblob**, is a basic rule-based model therefore it does not fully understand context, sarcasm, and sland words.
- The model only analyzes the post itself and ignores comments and replies, a crucial part of determining the sentiment of the post.
- Dataset is limited and outdated. This type of data should be constantly updated due to its fast changing trends.
- Many of the neutral posts are ambiguous

""")
st.subheader("Improvements")
st.markdown("""
            
- Implement more advanced **NLP models** such as **BERT** or **roBERTa** for more accurate sentiment classification
- Include more than just text like comment counts, upvotes, timestamps, etc
- Use **preprocessing** tools or slang dictionary to better indentify key indications

""")

st.header("Conclusion")
st.write("This project demonstrated how sentiment analysis can be used to interpret mental health-related posts on Reddit. By cleaning the data and applying **TextBlob**, we identified emotional trends across user content. While the model provides useful insights, it has limitations in context understanding and accuracy. Future improvements with advanced **NLP models** could lead to more reliable and meaningful analysis of online mental health discussions.")
