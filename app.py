import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Movie dataset.csv", encoding='latin1')
    df.dropna(subset=['Name', 'Year', 'Duration', 'Votes', 'Rating', 'Genre'], inplace=True)
    df.drop_duplicates(subset=['Name', 'Year', 'Director'], keep='first', inplace=True)
    df['Year'] = df['Year'].astype(str).str.strip('()').astype(int)
    df['Duration'] = df['Duration'].astype(str).str.replace(r'min', '').astype(int)
    df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(int)
    return df

df = load_data()

# Create a separate DataFrame for EDA to retain all columns
df_eda = df.copy()

# Feature engineering for model training
df.drop(['Name', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1, inplace=True)

# Splitting the data
X = df[['Year', 'Duration', 'Votes']]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=231)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=23),
    'SGD Regressor': SGDRegressor(max_iter=100, random_state=1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=231)
}

# Train models
def train_models():
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

trained_models = train_models()

# Streamlit UI
st.title('Movie Rating Prediction')
st.write('Enter the details to predict the movie rating:')

year = st.number_input('Year', min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=2000)
duration = st.number_input('Duration (in minutes)', min_value=int(df['Duration'].min()), max_value=int(df['Duration'].max()), value=120)
votes = st.number_input('Votes', min_value=int(df['Votes'].min()), max_value=int(df['Votes'].max()), value=1000)

model_name = st.selectbox('Select Model', list(models.keys()))

if st.button('Predict'):
    model = trained_models[model_name]
    pred = model.predict([[year, duration, votes]])
    st.write(f'The predicted rating for the movie is: {pred[0]:.2f}')

# Display R-squared scores
st.subheader('Model Performance')
r2_scores = {name: r2_score(y_test, model.predict(X_test)) for name, model in trained_models.items()}
st.write(r2_scores)

# EDA Section
st.subheader('Exploratory Data Analysis')

st.write('Number of Movies Released Each Year:')
yearly_movie_counts = df_eda['Year'].value_counts().sort_index()
st.bar_chart(yearly_movie_counts)

st.write('Number of Movies Released Per Genre:')
dummies = df_eda['Genre'].str.get_dummies(', ')
df_genre = pd.concat([df_eda, dummies], axis=1)
genre_movie_counts = df_genre[dummies.columns].sum().sort_index()
st.bar_chart(genre_movie_counts)

st.write('Top 20 Directors with the Most Movies:')
top_20_directors = df_eda['Director'].value_counts().head(20)
st.bar_chart(top_20_directors)

st.write('Top 20 Actors with the Most Movies:')
top_20_actors = df_eda['Actor 1'].value_counts().head(20)
st.bar_chart(top_20_actors)
