from flask import Flask, render_template, jsonify, request
import pandas as pd
import ast 
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process

app = Flask(__name__, template_folder='templates/html/')

df = pd.read_csv("cleaned_dataset.csv")
highest_grosing_movie_row = df.loc[df['Gross'].idxmax()]
lowest_grosing_movie_row = df.loc[df['Gross'].idxmin()]

lowest_rated_movie_row = df.loc[df['IMDB_Rating'].idxmin()]
highest_rated_movie_row = df.loc[df['IMDB_Rating'].idxmax()]


'''
Top 10 Genres Calculation Start
'''

genres = df['Genre'].apply(lambda x: ast.literal_eval(x))  # Convert string representation to list
genres = pd.Series([item for sublist in genres for item in sublist])  # Flatten the list
genre_counts = genres.value_counts().head(10)
genre_counts = pd.DataFrame({
    'Genre': genre_counts.index,
    'Frequency': genre_counts.values
})
'''
Top 10 Genres Calculation End
'''

'''
Doughnut Chart Start
'''
# Counting the frequency of each certificate
certificate_counts = df['Certificate'].value_counts()
'''
Doughnut Chart End
'''

'''
Recommendation system start
'''
dfi = pd.read_csv('imdb_top_1000.csv')
dfi['Genre'].fillna('', inplace=True)
dfi['Overview'].fillna('', inplace=True)
dfi = dfi.dropna()
dfi['Genre'] = dfi['Genre'].apply(lambda x: ' '.join(x.split(',')))
dfi['combined_features'] = dfi['Genre'] + ' ' + dfi['Overview'] 

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(dfi['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on similarity
def get_recommendations(query_title, cosine_sim=cosine_sim, titles=dfi['Series_Title']):
    # Perform fuzzy matching to find the closest matching movie titles
    matches = process.extract(query_title, titles, limit=5)
    closest_match = matches[0][0]  # Get the closest matching title
    idx = dfi[dfi['Series_Title'] == closest_match].index[0]  # Get the index of the closest match
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return dfi['Series_Title'].iloc[movie_indices].values.tolist()

@app.route('/wow')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.form['message']
    
    # Recommendation logic goes here
    if data.lower() in ['hi', 'hello', 'hey']:
        response = "Hello! How can I assist you today? Please enter a movie name and I'll recommend movies simalr to that genre"
    elif data.lower() in ['bye', 'goodbye']:
        response = "Goodbye! Have a great day!"
    else:
        try:
            recommendations = get_recommendations(data)
            response = "Here are some movie recommendations for you:\n" + "<br>".join(recommendations)
        except IndexError:
            response = "Sorry, I couldn't find any recommendations for that movie. Please try another one."
    return jsonify({'response': response})  # Return the response in JSON format

'''
Recommendation system start
'''

@app.route('/')
def index():
    average_gross = 5
    average_imdb_rating = 5
    average_rating = 5
    average_runtime = 5
    # Plotly chart
    fig = px.bar(genre_counts.head(10), 
                 x='Genre', 
                 y='Frequency', 
                 title='Frequency of Movie Genres',
                 labels={'Frequency': 'Frequency', 'Genre': 'Genre'},
                 color='Frequency',
                 color_continuous_scale=px.colors.sequential.Sunsetdark_r[::-1],
                 #color_continuous_scale=[[0, '#7f81ff'], [1, '#7f81ff']],                 
                 template='plotly_white')
    fig.update_layout(xaxis_tickangle=-45)

    # Convert the Plotly chart to JSON format
    plot_json = fig.to_json()

    fig1 = px.pie(names=certificate_counts.index, 
                 values=certificate_counts.values, 
                 hole=0.4, 
                 title='Distribution of Movie Certificates',
                 color_discrete_sequence=px.colors.sequential.Sunset[::-1])
    fig1.update_traces(textposition='inside', textinfo='percent+label')

    # Remove the legend
    fig1.update_layout(showlegend=True)

    # Convert the Plotly chart to JSON format
    doughnut = fig1.to_json()

    # Trend 1: IMDb Ratings Distribution
    trend1 = px.line(df.groupby('Released_Year')['Gross'].sum().reset_index(), 
                 x='Released_Year', 
                 y='Gross', 
                 title='Total Gross Earnings Over the Years',
                 labels={'Released_Year': 'Year', 'Gross': 'Total Gross Earnings ($)'},
                 line_shape='spline', 
                 color_discrete_sequence=px.colors.sequential.Sunsetdark_r) 
    # Adjust layout for transparent background
    trend1.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    trend1_plot = trend1.to_json()

    return render_template('index.html', title='Welcome', name='World', highest_grosing_movie_row = highest_grosing_movie_row,
                           lowest_grosing_movie_row=lowest_grosing_movie_row, plot_json=plot_json, doughnut=doughnut, trend1_plot=trend1_plot,
                           highest_rated_movie_row=highest_rated_movie_row, lowest_rated_movie_row=lowest_rated_movie_row, average_gross=average_gross, average_imdb_rating=average_imdb_rating,
                           average_rating=average_rating, average_runtime=average_runtime)

if __name__ == '__main__':
    app.run(debug=True)
