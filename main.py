from flask import Flask, render_template, jsonify
import pandas as pd
import ast 
import plotly.express as px

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

@app.route('/')
def index():
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
    fig1.update_layout(showlegend=False)

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
                           highest_rated_movie_row=highest_rated_movie_row, lowest_rated_movie_row=lowest_rated_movie_row)

if __name__ == '__main__':
    app.run(debug=True)
