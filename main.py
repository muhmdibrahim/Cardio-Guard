import MySQLdb.cursors
import re, pickle
from blog import *

with open('RF.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

<<<<<<< HEAD
=======
app = Flask(__name__)
model = pickle.load(open('RF_C.pkl','rb'))

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e6d7g8h9i10'

# Enter your database connection details below
app.config['MYSQL_HOST'] = ''
app.config['MYSQL_USER'] = 'hemaa_ai'
app.config['MYSQL_PASSWORD'] = 'hemaa2468' #Replace ******* with  your database password.
app.config['MYSQL_DB'] = 'login_users'

# Intialize MySQL
mysql = MySQL(app)


# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
>>>>>>> aaab8a68d745db34306165969746f9f8ca886370
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))

        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['email'] = account['email']
            session['status'] = account['status']
            
            return redirect(url_for('home'))
        else:
            flash("Incorrect username/password!", "danger")
    return render_template('auth/login.html',title="Login")


# http://localhost:5000/pythonlogin/register 
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        profile_picture = request.files['profile_picture']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute( "SELECT * FROM accounts WHERE username = %s", (username,) )
        account = cursor.fetchone()

        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s, NULL, %s)', (username, password, email, profile_picture, ))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))

    elif request.method == 'POST':
        flash("Please fill out the form!", "danger")
    return render_template('auth/register.html',title="Register")

# http://localhost:5000/pythinlogin/home 
# check session for each page

@app.route('/')
def home():
    if 'loggedin' in session:
        return render_template('home/home.html', username=session['username'],title="Home")
    return redirect(url_for('login'))  

@app.route('/profile')
def profile():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if 'loggedin' in session:
        cursor.execute("SELECT * FROM accounts WHERE username = %s", (session['username'], ))
        account = cursor.fetchone()
        return render_template('auth/profile.html', account = account, title="Profile")
    
    return redirect(url_for('login'))  

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))

@app.route('/hdd')
def hdd():
    if 'loggedin' in session:
        return render_template('result/hdd.html', username=session['username'],title="hdd")
    return redirect(url_for('login'))

@app.route('/healthyvsunhealthyfood')
def healthyvsunhealthyfood():
    if 'loggedin' in session:
        return render_template('home/food_home.html', username=session['username'],title="Healthy vs Unhealthy Food")
    return redirect(url_for('login'))

@app.route('/healthyvsunhealthydrinks')
def healthyvsunhealthydrinks():
    if 'loggedin' in session:
        return render_template('CardioGuardBot/index.html', username=session['username'],title="Healthy vs Unhealthy Drinks")
    return redirect(url_for('login'))

@app.route('/result')
def result():
    if 'loggedin' in session:
        return render_template('result/result.html', username=session['username'], title="predict")
    return redirect(url_for('login'))

# predict HDD page
import numpy as np

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        chest_pain_type = request.form.get('chest pain type')
        resting_bp_s = int(request.form['resting bp s']) 
        cholesterol = int(request.form['cholesterol'])
        fasting_blood_sugar = request.form.get('fasting_blood_sugar')
        resting_ecg = request.form.get('resting ecg')
        max_heart_rate = int(request.form['max heart rate'])
        exercise_angina = request.form.get('exercise angina')
        oldpeak = int(request.form['oldpeak'])
        ST_slope = request.form.get('ST slope')
        
        data = np.array([[age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,
                          resting_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope]])
        data = data.astype(int)
        prediction = model.predict(data)[0]

        if int(prediction) == 0:
            status = "good health"
        else:
            status = "patient"

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        if 'loggedin' in session:
            cursor.execute( "UPDATE accounts SET status = %s WHERE username = %s", (status, session['username']))
            mysql.connection.commit()

        return render_template('result/result.html', prediction=prediction , username=session['username'])

# Food Detection: Two FUNCTIONS TO DETECT (PREDICT_IMG, DISPLAY)
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
from ultralytics import YOLO
import requests
import json

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your model here
model2 = YOLO("best.pt")

names = ['Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon', 'alooparatha-chapati', 'avocado', 'beans', 'beet', 'bell pepper', 'besan_cheela', 'biriyani', 'broccoli', 'brus capusta', 'cabbage', 'carrot', 'carrot_eggs', 'cayliflower', 'celery', 'chicken_nuggets', 'chinese_cabbage', 'chinese_sausage', 'chole', 'corn', 'cucumber', 'curry', 'dal', 'dalmakhani', 'dosa', 'dosa-uttapam', 'eggplant', 'fasol', 'fried_chicken', 'fried_dumplings', 'fried_eggs', 'garlic', 'gulab_jamun', 'gulabjamun', 'hot pepper', 'idli', 'khichdi', 'mung_bean_sprouts', 'omelette', 'onion', 'palak_paneer', 'palakpaneer', 'papad', 'peas', 'plainrice', 'poha', 'poori', 'potato', 'pumpkin', 'rajma', 'rasgulla', 'rediska', 'redka', 'rice', 'salad', 'sambhar', 'samosa', 'squash-patisson', 'tomato', 'triangle_hash_brown', 'vada', 'vegetable marrow', 'water_spinach']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def get_unique_class(numbers):
    # Convert the list to a set to remove duplicates, then back to a list
    unique_numbers = list(set(numbers))
    return unique_numbers

def convert_to_lowercase(word_list):
    return [word.lower() for word in word_list]

healthy_foods = [
    'Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon',
    'Avocado', 'Beans', 'Beet', 'Bell pepper', 'Broccoli', 'Brussels sprouts',
    'Cabbage', 'Carrot', 'Cauliflower', 'Celery', 'Chinese cabbage', 'Corn',
    'Cucumber', 'Dal', 'Eggplant', 'Garlic', 'Hot pepper', 'Idli', 'Khichdi',
    'Mung bean sprouts', 'Onion', 'Peas', 'Pumpkin', 'Rajma', 'Radish',
    'Salad', 'Sambhar', 'Squash', 'Tomato', 'Vegetable marrow', 'Water spinach',
]

unhealthy_foods = [
    'Alooparatha-chapati', 'Besan cheela', 'Biriyani', 'Carrot eggs',
    'Chicken nuggets', 'Chinese sausage', 'Chole', 'Curry', 'Dal makhani',
    'Dosa', 'Dosa-uttapam', 'Fried chicken', 'Fried dumplings', 'Fried eggs',
    'Gulab jamun', 'Gulabjamun', 'Omelette', 'Palak paneer', 'Papad',
    'Plain rice', 'Poha', 'Poori', 'Rasgulla', 'Rice', 'Samosa',
    'Triangle hash brown', 'Vada'
]


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image and get prediction
            image = Image.open(filepath)
            res = model2.predict(source=image, conf=0.25 )

            lst = []
            for r in res:
                lst.extend(list(map(int, r.boxes.cls.tolist())))

            lst = get_unique_class(lst)

            string_all_food = " , ".join([names[i] for i in lst])
            query = string_all_food
            api_url = f'https://api.api-ninjas.com/v1/nutrition?query={query}'
            response = requests.get(api_url, headers={'X-Api-Key': 'IITQvHlB7YnyNrn15o9GSQ==Z34Dl3yNjVHOD0bn'})

            # Parse the JSON text into Python objects (list of dictionaries)
            foods = json.loads(response.text)

            nutritional_info = []
            nutritional_info.append(f"In This Image There are {string_all_food}")

            prediction =" "
            for i in lst:
                if names[i] in healthy_foods:
                    prediction = f"Yummy, You can eat it😋\n"
                elif names[i] in unhealthy_foods:
                    prediction = f"It's Better to avoid this food🤔\n"

            # Gather the nutritional information for each food item
            for food in foods:
                info = {
                    "name": food['name'],
                    "fat_total": food['fat_total_g'],
                    "fat_saturated": food['fat_saturated_g'],
                    "sodium": food['sodium_mg'],
                    "potassium": food['potassium_mg'],
                    "cholesterol": food['cholesterol_mg'],
                    "carbohydrates_total": food['carbohydrates_total_g'],
                    "fiber": food['fiber_g'],
                    "sugar": food['sugar_g']
                }
                nutritional_info.append(info)

            return render_template('result/food_result.html', prediction=prediction, nutritional_info=nutritional_info, image_url=filepath, username=session['username'])
    return render_template('home/food_home.html')
    
# Symptoms
#################################################################################
import joblib
Selected_Symptoms_Model = joblib.load('Selected_Symptoms_Model.pkl')

symptoms = ['itching', 'skin_rash', 'continuous_sneezing', 'chills', 'joint_pain',
        'stomach_pain', 'acidity', 'vomiting', 'burning_micturition', 'fatigue',
        'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'cough',
        'high_fever', 'breathlessness', 'sweating', 'indigestion', 'headache',
        'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
        'mild_fever', 'yellowing_of_eyes', 'swelled_lymph_nodes', 'malaise',
        'blurred_and_distorted_vision', 'phlegm', 'chest_pain',
        'fast_heart_rate', 'neck_pain', 'dizziness', 'obesity',
        'excessive_hunger', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
        'loss_of_balance', 'depression', 'irritability', 'muscle_pain',
        'red_spots_over_body', 'abnormal_menstruation', 'painful_walking']
import pandas as pd
data_symptoms = pd.read_csv("Data.csv")

selected_dict_symptoms = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer disease',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'
}

@app.route('/selectsymptoms')
def select_symptoms():
    if 'loggedin' in session:
        return render_template('home/symptoms.html', username=session['username'],title="Predict Selected Symptoms", symptoms=symptoms)
    return redirect(url_for('login'))

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    # Initialize feature vector with zeros
    features = []

    # Get feature values from form data
    for i , f in enumerate(symptoms):
        feature_name = f
        if feature_name in request.form:
            features.append(1)
        else:
            features.append(0)

    # Reshape features for prediction
    features = np.array(features)
    features = features.astype(int)
    features = features.reshape(1, -1)

    # Make prediction using the model
    prediction = Selected_Symptoms_Model.predict(features)[0]

    # Determine prediction result
    disease_name = selected_dict_symptoms[prediction]
    description = data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Description'].values[0]
    diets = data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Diet'].values[0] 
    workout = data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'workout'].values[0]
    Medication = data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Medication'].values[0]
    Precaution = data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Precaution_1'].values[0]
    Precaution += " , " + data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Precaution_2'].values[0]
    Precaution += " , " + data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Precaution_3'].values[0]
    Precaution += " And " + data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Precaution_4'].values[0]

    # Create result table data
    result_table = {
        'Disease': disease_name,
        'Description': description,
        'Diets': diets,
        'workout': workout,
        'Medication': Medication,
        'Precaution': Precaution
    }

    return render_template('home/symptoms.html', prediction_result=result_table)

#symptoms from text
from proceesing_text_helper import *

@app.route('/select_symptoms_text')
def select_symptoms_text():
    if 'loggedin' in session:
        return render_template('home/symptoms_text.html', username=session['username'],title="Predict Text Symptoms", error_message=None)
    return redirect(url_for('login'))

# Load the SVM model
svm_model = joblib.load("Symptoms_To_Disease.pkl")

# Dictionary mapping for disease names
dict_symptoms = {
    0: 'Acne',
    1: 'Arthritis',
    2: 'Bronchial Asthma',
    3: 'Cervical spondylosis',
    4: 'Chicken pox',
    5: 'Common Cold',
    6: 'Dengue',
    7: 'Dimorphic hemmorhoids(piles)',
    8: 'Fungal infection',
    9: 'Hypertension',
    10: 'Impetigo',
    11: 'Jaundice',
    12: 'Malaria',
    13: 'Migraine',
    14: 'Pneumonia',
    15: 'Psoriasis',
    16: 'Typhoid',
    17: 'Varicose veins',
    18: 'Allergy',
    19: 'Diabetes',
    20: 'Drug Reaction',
    21: 'Gastroenteritis',
    22: 'Peptic ulcer disease',
    23: 'Urinary tract infection'
}

# Helper function to format lists from CSV
def format_list(input_text):
    return ', '.join(eval(input_text))

@app.route('/process_text_symptoms', methods=['POST'])
def process_text_symptoms():
    text = request.form['input_text'].strip()  # Get input text and remove leading/trailing whitespace

    if not text:
        return render_template('index.html', error_message="Nothing has been written, please write what you feel")

    text = preprocessing(pd.Series(text))  # Preprocess text as needed

    prediction = svm_model.predict(text)[0]  # Make prediction

    disease_name = dict_symptoms.get(prediction, 'Unknown Disease')

    description = data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Description'].values[0]
    diets = format_list(data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Diet'].values[0])
    workout = format_list(data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'workout'].values[0])
    medication = format_list(data_symptoms.loc[data_symptoms['Disease'] == disease_name, 'Medication'].values[0])
    precaution = ', '.join(
        data_symptoms.loc[data_symptoms['Disease'] == disease_name, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0])

    result_table = {
        'Disease': disease_name,
        'Description': description,
        'Diets': diets,
        'Workout': workout,
        'Medication': medication,
        'Precaution': precaution
    }

    return render_template('home/symptoms_text.html', username=session['username'],title="Predict Text Symptoms", prediction_result=result_table)

if __name__ =='__main__':
	app.run(debug=True)
