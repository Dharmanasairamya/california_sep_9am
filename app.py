from flask import Flask, request, render_template
import joblib

# Load model
obj = joblib.load('california.joblib')

model = obj['MODEL']
columns = obj['Columns']

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html', columns=columns)


@app.route('/predict', methods=['POST'])
def predict():

    input_data = []

    for column in columns:
        value = float(request.form[column])
        input_data.append(value)

    prediction = model.predict([input_data])[0]

    return render_template(
        'index.html',
        columns=columns,
        prediction=prediction
    )


if __name__ == '__main__':
    app.run(debug=True)