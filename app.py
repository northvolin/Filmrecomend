import pandas as pd

# Загрузим данные MovieLens
try:
 ratings = pd.read_csv('ratings.csv')
 print("Ratings columns:", ratings.columns)
except FileNotFoundError:
 print("File ratings.csv not found.")
 ratings = pd.DataFrame()  # или другой способ обработки ошибки

try:
 movies = pd.read_csv('movies.csv')
 print("Movies columns:", movies.columns)
except FileNotFoundError:
 print("File movies.csv not found.")
 movies = pd.DataFrame()  # или другой способ обработки ошибки

# Проверьте, что нужные столбцы присутствуют
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
 return render_template('index.html', movies=movies.to_dict(orient='records'))

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
