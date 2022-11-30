import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

# Data import:
music_data = pd.read_csv('music.cvs')
print(music_data)

# Data processing:
x = music_data.drop(columns=['genre'])
print(x)
y = music_data['genre']
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Decision tree model process:
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Model precision measure:
predictions_test = model.predict(x_test, y_test)
score = accuracy_score(y_test, predictions_test)
print(score)

# Prediction example:
predictions = model.predict([21, 1], [22, 0])
print(predictions)


# Store the trained model in a file, so we don't need to train it anymore:
joblib.dump(model, "music-recommender.joblib")
# Load the saved model:
joblib.load("music-recommender.joblib")
# And them we can simply process the predictions:
predictions = model.predict([21, 1], [22, 0])


# Upload the decision tree on a file, so it can be visualized:
tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)

