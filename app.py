from flask import Flask, request, render_template
import pandas as pd
from surprise import Dataset, Reader, SVD

app = Flask(__name__)

def load_data():
    try:
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        print("File ratings.csv not found. Please ensure the file is in the correct directory.")
        ratings = pd.DataFrame()  # или другой способ обработки ошибки

    try:
        movies = pd.read_csv('movies.csv')
    except FileNotFoundError:
        print("File movies.csv not found. Please ensure the file is in the correct directory.")
        movies = pd.DataFrame()  # или другой способ обработки ошибки

    return ratings, movies

ratings, movies = load_data()

# Проверяем, что нужные столбцы присутствуют
required_columns = ['userId', 'movieId', 'rating']
if all(column in ratings.columns for column in required_columns):
    # Создаем объект Reader
    reader = Reader(rating_scale=(0.5, 5.0))

    # Загружаем данные в формат Surprise
    data = Dataset.load_from_df(ratings[required_columns], reader)

    # Создаем и обучаем модель SVD
    model = SVD()
    trainset = data.build_full_trainset()
    model.fit(trainset)
else:
    print(f"Ratings file does not contain required columns: {required_columns}")
    data = None
    model = None

@app.route('/')
def index():
    if not movies.empty:
        return render_template('index.html', movies=movies.to_dict(orient='records'))
    else:
        return "Movies data not available."

@app.route('/recommend', methods=['POST'])
def recommend():
    user_ratings = request.form.getlist('rating')
    user_id = int(request.form['user_id'])
    user_ratings = [float(rating) for rating in user_ratings]

    user_ratings_dict = {movie_id: rating for movie_id, rating in zip(movies['movieId'], user_ratings)}

    predictions = []
    if model:
        for movie_id in movies['movieId']:
            if movie_id not in user_ratings_dict:
                predictions.append((movie_id, model.predict(user_id, movie_id).est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_movies = [movies[movies['movieId'] == movie_id]['title'].values[0] for movie_id, _ in predictions[:10]]
    else:
        recommended_movies = []

    return render_template('recommend.html', movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
