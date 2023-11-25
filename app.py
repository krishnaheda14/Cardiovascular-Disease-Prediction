from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)
rf_model = RandomForestClassifier(random_state=42)

#Load and preprocess the heart disease dataset
heart_dataset = pd.read_csv('cardiovascular disease data.csv')
le = LabelEncoder()
for col in heart_dataset.columns:
    if heart_dataset[col].dtype == 'object':
        heart_dataset[col] = le.fit_transform(heart_dataset[col])

X = heart_dataset.drop(columns=['target'])
Y = heart_dataset['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and fit the StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X)

standardScaler = StandardScaler()
standardScaler.fit(X_train)

# Fit the models during initialization
rf_model.fit(X_train, Y_train)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])


        # Scale the input data
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        input_data_scaled = standardScaler.transform(input_data)

        # Make predictions using the loaded models

        result = rf_model.predict(input_data_scaled)[0]



        return render_template('result.html', result_message=result)

if __name__ == '__main__':
    app.run(debug=True)
