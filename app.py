from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a KNN classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get feature inputs from the user
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']

    # Convert feature inputs to floats
    features = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]

    # Predict the species of the flower using the trained classifier
    species = clf.predict([features])[0]

    # Return the predicted species
    return render_template('predict.html', species=iris.target_names[species])


if __name__ == '__main__':
    app.run(debug=True)
