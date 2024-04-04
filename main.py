from flask import Flask, render_template, jsonify, request
import pandas as pd
import ast 
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import tempfile

app = Flask(__name__, template_folder='templates/html/')


def create_certificate_doughnut_chart(df):
    certificate_counts = df['Certificate'].value_counts()
    
    fig = px.pie(names=certificate_counts.index, 
                 values=certificate_counts.values, 
                 hole=0.4, 
                 title='Distribution of Movie Certificates',
                 color_discrete_sequence=px.colors.sequential.Sunset[::-1])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    
    return fig.to_json()

def create_gross_earnings_trend(df):
    trend = px.line(df.groupby('Released_Year')['Gross'].sum().reset_index(), 
                    x='Released_Year', 
                    y='Gross', 
                    title='Total Gross Earnings Over the Years',
                    labels={'Released_Year': 'Year', 'Gross': 'Total Gross Earnings ($)'},
                    line_shape='spline', 
                    color_discrete_sequence=px.colors.sequential.Sunsetdark_r) 
    trend.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    
    return trend.to_json()


def create_imdb_vs_gross_scatter(df):
    data = go.Scatter(
        x=df['IMDB_Rating'],
        y=df['Gross'],
        mode='markers',
        marker=dict(
            color=px.colors.sequential.Sunset_r,
            size=8
        )
    )

    layout = go.Layout(
        title='IMDb Ratings vs. Gross Earnings',
        xaxis=dict(title='IMDb Rating'),
        yaxis=dict(title='Gross Earnings'),
        hovermode='closest',
        plot_bgcolor='rgba(253, 245, 230, 1)',  # Sunset theme background color
    )

    fig = go.Figure(data=[data], layout=layout)
    
    return fig.to_json()

def create_imdb_ratings_bar_chart(df):
    fig = px.bar(df, x='Series_Title', y='IMDB_Rating', hover_data=['Released_Year', 'Director'],
                 title='IMDb Ratings of Movies', labels={'IMDB_Rating': 'IMDb Rating'})
    fig.update_layout(xaxis_title='Movie Title', yaxis_title='IMDb Rating', xaxis_tickangle=-45,
                      title_x=0.5, title_font_size=24)
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_average_imdb_ratings_by_director_bar_chart(df):
    avg_ratings_by_director = df.groupby('Director')['IMDB_Rating'].mean().reset_index()
    avg_ratings_by_director = avg_ratings_by_director.sort_values(by='IMDB_Rating', ascending=False).head(10)
    
    fig = px.bar(avg_ratings_by_director, x='Director', y='IMDB_Rating', 
                 title='Average IMDb Ratings by Director',
                 labels={'Director': 'Director', 'IMDB_Rating': 'Average IMDb Rating'},
                 color='IMDB_Rating',
                 color_continuous_scale=px.colors.sequential.Sunsetdark_r)
    
    # Adjusting y-axis range dynamically
    min_rating = avg_ratings_by_director['IMDB_Rating'].min()
    max_rating = avg_ratings_by_director['IMDB_Rating'].max()
    y_range = [min_rating - 0.5, max_rating + 0.5]  # Adjust the padding as needed
    
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=y_range))
    return fig.to_json()


def create_genre_bar_chart(df):
    genres = df['Genre'].apply(lambda x: ast.literal_eval(x))
    genres = pd.Series([item for sublist in genres for item in sublist])
    genre_counts = genres.value_counts().head(10)
    genre_counts = pd.DataFrame({
        'Genre': genre_counts.index,
        'Frequency': genre_counts.values
    })
    
    fig = px.bar(genre_counts.head(10), 
                 x='Genre', 
                 y='Frequency', 
                 title='Frequency of Movie Genres',
                 labels={'Frequency': 'Frequency', 'Genre': 'Genre'},
                 color='Frequency',
                 color_continuous_scale=px.colors.sequential.Sunsetdark_r[::-1],
                 template='plotly_white')
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig.to_json()

def create_movie_release_year_imdb_rating_treemap(df):
    avg_imdb_ratings_by_year = df.groupby('Released_Year')['IMDB_Rating'].mean().reset_index()
    
    # Create a DataFrame for the treemap with 'Released_Year' as the path and 'IMDB_Rating' as the values
    treemap_df = pd.DataFrame({
        'release_year': avg_imdb_ratings_by_year['Released_Year'],
        'rating': avg_imdb_ratings_by_year['IMDB_Rating']
    })
    
    # Create the treemap figure
    fig = px.treemap(treemap_df, path=['release_year'], values='rating', 
                     title='Movie Release Years and IMDb Ratings',
                     color='rating', 
                     color_continuous_scale=px.colors.sequential.Sunsetdark_r[::-1])
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    return fig.to_json()


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
    df = pd.read_csv("cleaned_dataset.csv")
    
    highest_grosing_movie_row = df.loc[df['Gross'].idxmax()]
    lowest_grosing_movie_row = df.loc[df['Gross'].idxmin()]
    lowest_rated_movie_row = df.loc[df['IMDB_Rating'].idxmin()]
    highest_rated_movie_row = df.loc[df['IMDB_Rating'].idxmax()]
    
    average_gross = round(df['Gross'].mean(), 2)
    average_imdb_rating = round(df['IMDB_Rating'].mean(), 2)
    average_rating = round(df['Meta_score'].mean(), 2)
    average_runtime = round(df['Runtime'].mean(), 2)
    
    plot_json = create_genre_bar_chart(df)
    doughnut = create_certificate_doughnut_chart(df)
    trend1_plot = create_gross_earnings_trend(df)
    graph = create_imdb_vs_gross_scatter(df)
    plot_html = create_imdb_ratings_bar_chart(df)
    director_imdb = create_average_imdb_ratings_by_director_bar_chart(df)
    teremap = create_movie_release_year_imdb_rating_treemap(df)
    
    return render_template('index.html', title='Welcome', name='World', highest_grosing_movie_row=highest_grosing_movie_row,
                           lowest_grosing_movie_row=lowest_grosing_movie_row, plot_json=plot_json, doughnut=doughnut,
                           trend1_plot=trend1_plot, highest_rated_movie_row=highest_rated_movie_row,
                           lowest_rated_movie_row=lowest_rated_movie_row, average_gross=average_gross,
                           average_imdb_rating=average_imdb_rating, average_rating=average_rating,
                           average_runtime=average_runtime, graph=graph, plot_html=plot_html,
                           director_imdb=director_imdb, teremap=teremap)


if __name__ == '__main__':
    app.run(debug=True)
