

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('happiness_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    happiness_rank = float(request.form['Happiness Rank'])
    lower_ci = float(request.form['Lower Confidence Interval'])
    upper_ci = float(request.form['Upper Confidence Interval'])
    gdp_per_capita = float(request.form['Economy (GDP per Capita)'])
    family = float(request.form['Family'])
    life_expectancy = float(request.form['Health (Life Expectancy)'])
    freedom = float(request.form['Freedom'])
    government_corruption = float(request.form['Trust (Government Corruption)'])
    generosity = float(request.form['Generosity'])
    dystopia_residual = float(request.form['Dystopia Residual'])

    final_features = [[happiness_rank, lower_ci, upper_ci, gdp_per_capita, family, life_expectancy, freedom, government_corruption, generosity, dystopia_residual]]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Happiness Score: {}'.format(output))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
