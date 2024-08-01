from flask import Flask, request, render_template
import pandas as pd
from surprise import Dataset, Reader, SVD

app = Flask(__name__)

# Загрузим данные MovieLens
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Создаем объект Reader
reader = Reader(rating_scale=(0.5, 5.0))

# Загружаем данные в формат Surprise
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Создаем и обучаем модель SVD
model = SVD()
trainset = data.build_full_trainset()
model.fit(trainset)

@app.route('/')
def index():
 return render_template('index.html', movies=movies.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def recommend():
 user_ratings = request.form.getlist('rating')
 user_id = int(request.form['user_id'])
 user_ratings = [float(rating) for rating in user_ratings]

 user_ratings_dict = {movie_id: rating for movie_id, rating in zip(movies['movieId'], user_ratings)}

 predictions = []
 for movie_id in movies['movieId']:
  if movie_id not in user_ratings_dict:
   predictions.append((movie_id, model.predict(user_id, movie_id).est))

 predictions.sort(key=lambda x: x[1], reverse=True)
 recommended_movies = [movies[movies['movieId'] == movie_id]['title'].values[0] for movie_id, _ in predictions[:10]]

 return render_template('recommend.html', movies=recommended_movies)

if __name__ == '__main__':
 app.run(debug=True)
