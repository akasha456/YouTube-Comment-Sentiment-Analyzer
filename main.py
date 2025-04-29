import os
import pandas as pd
import re
from googleapiclient.discovery import build
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# ✅ YouTube API Setup
API_KEY = 'AIzaSyD0q1NIvX9vTKiDJI5a2rtCpQvlNGFAUeM'
youtube = build('youtube', 'v3', developerKey=API_KEY)

# ✅ Load and Prepare Training Data
data = pd.read_csv('YoutubeCommentsDataSet.csv')
data = data.dropna(subset=['Comment', 'Sentiment'])

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['CleanComment'] = data['Comment'].apply(preprocess_text)

# Map sentiment labels to numeric
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['Label'] = data['Sentiment'].map(label_mapping)

X = data['CleanComment'].tolist()
y = data['Label'].tolist()

# ✅ Improved Model Pipeline (SVM)
classifier_model = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2), stop_words='english')),
    ('clf', LinearSVC(class_weight='balanced', max_iter=10000))
])
classifier_model.fit(X, y)

# ✅ Helper Functions
def get_video_id_from_url(video_url):
    video_id = None
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1].split('&')[0]
    elif 'youtu.be' in video_url:
        video_id = video_url.split('/')[-1]
    return video_id.split('?')[0] if video_id else None

def get_video_comments(video_id):
    comments = []
    try:
        results = youtube.commentThreads().list(
            part='snippet', videoId=video_id, textFormat='plainText', maxResults=100
        ).execute()
        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            results = youtube.commentThreads().list(
                part='snippet', videoId=video_id, textFormat='plainText',
                pageToken=results.get('nextPageToken', ''), maxResults=100
            ).execute() if 'nextPageToken' in results else None
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []
    return comments

def classify_comments(comments):
    categorized_comments = {'good': [], 'bad': [], 'neutral': []}
    for comment in comments:
        cleaned = preprocess_text(comment)
        model_output = classifier_model.predict([cleaned])[0]
        category = 'good' if model_output == 2 else 'bad' if model_output == 0 else 'neutral'
        categorized_comments[category].append(comment)
    return categorized_comments

def plot_interactive_donut_chart(categorized_comments):
    labels = ['Good', 'Bad', 'Neutral']
    sizes = [len(categorized_comments['good']), len(categorized_comments['bad']), len(categorized_comments['neutral'])]
    colors = ['#00ff00', '#ff0000', '#ffff00']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        hole=0.4,
        marker=dict(colors=colors),
        hoverinfo='label+percent',
        textinfo='value+percent',
        textfont_size=15
    )])

    fig.update_layout(
        title_text='Sentiment Distribution of YouTube Comments',
        annotations=[dict(text='YCSA', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig)

# ✅ Streamlit App UI
def main():
    st.title("YouTube Comment Sentiment Analyzer")
    st.markdown("<hr style='height:5px;border:none;background:linear-gradient(to right, red,orange,yellow,green,blue,indigo,violet);'>", unsafe_allow_html=True)

    video_url = st.text_input("Enter the YouTube video URL:")
    if st.button("Analyze") and video_url:
        video_id = get_video_id_from_url(video_url)
        if not video_id:
            st.error("Invalid YouTube URL.")
            return
        comments = get_video_comments(video_id)
        categorized_comments = classify_comments(comments)

        if categorized_comments:
            st.subheader(f"Total Comments: {len(comments)}")
            plot_interactive_donut_chart(categorized_comments)
            for sentiment, comment_list in categorized_comments.items():
                st.subheader(f"{sentiment.capitalize()} Comments:")
                st.markdown("\n".join([f"- {c}" for c in comment_list]) if comment_list else f"No {sentiment} comments found.")

if __name__ == "__main__":
    main()
