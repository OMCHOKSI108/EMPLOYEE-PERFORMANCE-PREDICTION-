from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load models and preprocessing artifacts with error handling
try:
    model_xg = pickle.load(open('backend/model_xg.pkl', 'rb'))  # XGBoost model
    model_rf = pickle.load(open('backend/model_RF.pkl', 'rb'))  # Random Forest model
    mappings = pickle.load(open('backend/mappings.pkl', 'rb'))
    target_scaler = pickle.load(open('backend/target_scaler.pkl', 'rb'))
    feature_scaler = pickle.load(open('backend/feature_scaler.pkl', 'rb'))
    feature_names = pickle.load(open('backend/feature_names.pkl', 'rb'))  # Load expected feature names
except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}. Please run the training script to generate all required files in 'backend/'.")
    exit()
except Exception as e:
    print(f"Error loading files: {str(e)}")
    exit()

# Define options for dropdowns
GENDERS = list(mappings['Gender'].keys())
EDU_BACKGROUNDS = list(mappings['EducationBackground'].keys())
MARITAL_STATUSES = list(mappings['MaritalStatus'].keys())
DEPARTMENTS = list(mappings['EmpDepartment'].keys())
JOB_ROLES = list(mappings['EmpJobRole'].keys())
TRAVEL_FREQUENCIES = list(mappings['BusinessTravelFrequency'].keys())
OVERTIME_OPTIONS = list(mappings['OverTime'].keys())
ATTRITION_OPTIONS = list(mappings['Attrition'].keys())
MODEL_OPTIONS = ['XGBoost', 'Random Forest']

def generate_bar_chart(data_dict, employee_name):
    plt.figure(figsize=(10, 6), facecolor='#f7f7f7')
    keys = ['Age', 'DistanceFromHome', 'EmpHourlyRate', 'TotalWorkExperienceInYears']
    values = [float(data_dict.get(k, 0)) for k in keys]
    bars = plt.bar(keys, values, color='#1f77b4', edgecolor='black', linewidth=1.2)
    plt.title(f'{employee_name} - Key Metrics', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.1f}", ha='center', fontsize=10)
    plt.tight_layout()
    filepath = f'static/charts/{employee_name}/bar_chart.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath

def generate_satisfaction_bar_chart(data_dict, employee_name):
    plt.figure(figsize=(10, 6), facecolor='#f7f7f7')
    keys = ['EmpJobSatisfaction', 'EnvSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction']
    values = [float(data_dict.get(k, 0)) for k in keys]
    bars = plt.barh(keys, values, color='#17becf', edgecolor='black', linewidth=1.2)
    plt.title(f'{employee_name} - Satisfaction Metrics', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Score (1-4)', fontsize=14)
    plt.xlim(0, 4)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va='center', ha='left', fontsize=10)
    plt.tight_layout()
    filepath = f'static/charts/{employee_name}/satisfaction_bar_chart.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath

def generate_gauge_chart(prediction, employee_name):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#f7f7f7')
    angles = np.linspace(0, np.pi, 100)
    x, y = np.cos(angles), np.sin(angles)
    ax.plot(x, y, color='#ddd', linewidth=12)
    angle = np.pi * (prediction - 1) / 99
    ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color='#ff6b6b', linewidth=16)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, -0.2, f'{int(prediction)}', ha='center', fontsize=24, fontweight='bold', color='#ff6b6b')
    ax.set_title(f'{employee_name} - Performance Score', fontsize=18, fontweight='bold', pad=15)
    ax.set_facecolor('#f0f0f0')
    filepath = f'static/charts/{employee_name}/gauge_chart.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath

def generate_line_chart(data_dict, employee_name):
    plt.figure(figsize=(10, 6), facecolor='#f7f7f7')
    keys = ['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    values = [float(data_dict.get(k, 0)) for k in keys]
    plt.plot(keys, values, marker='o', color='#9467bd', linewidth=2.5, markersize=10, label='Experience')
    plt.title(f'{employee_name} - Experience Timeline', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Years', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=12)
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center', fontsize=10)
    plt.tight_layout()
    filepath = f'static/charts/{employee_name}/line_chart.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath

def generate_categorical_bar_chart(data_dict, employee_name):
    plt.figure(figsize=(10, 6), facecolor='#f7f7f7')
    keys = ['Gender', 'OverTime', 'Attrition']
    values = [1 if data_dict.get(k) == 'Male' else 0 if k == 'Gender' else 1 if data_dict.get(k) == 'Yes' else 0 for k in keys]
    labels = ['Male' if values[0] == 1 else 'Female', 'Yes' if values[1] == 1 else 'No', 'Yes' if values[2] == 1 else 'No']
    bars = plt.bar(keys, values, color='#ff9999', edgecolor='black', linewidth=1.2)
    plt.title(f'{employee_name} - Categorical Breakdown', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Value (0 or 1)', fontsize=14)
    plt.xticks(ticks=range(len(keys)), labels=labels, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{int(yval)}", ha='center', fontsize=10)
    plt.tight_layout()
    filepath = f'static/charts/{employee_name}/categorical_bar_chart.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath

def generate_comparison_chart(prediction, employee_name, job_role):
    plt.figure(figsize=(10, 6), facecolor='#f7f7f7')
    company_avg, top_performer = 70, 85  # Mocked values
    bars = plt.bar(['Your Score', 'Company Avg', 'Top Performer'], [prediction, company_avg, top_performer], 
                   color=['#ff6b6b', '#66b3ff', '#99ff99'], edgecolor='black', linewidth=1.2)
    plt.title(f'{employee_name} - Performance Comparison', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Performance Score (1-100)', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{int(yval)}", ha='center', fontsize=10)
    plt.tight_layout()
    filepath = f'static/charts/{employee_name}/comparison_chart.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath

def get_insights(prediction, data_dict):
    if prediction < 40:
        level, color = "Low", "red"
    elif prediction <= 70:
        level, color = "Moderate", "yellow"
    else:
        level, color = "High", "green"
    
    strengths = sorted([(k, v) for k, v in data_dict.items() if isinstance(v, (int, float))], key=lambda x: x[1], reverse=True)[:2]
    weaknesses = sorted([(k, v) for k, v in data_dict.items() if isinstance(v, (int, float))], key=lambda x: x[1])[:2]
    
    recommendations = []
    if data_dict.get('EmpWorkLifeBalance', 0) < 2:
        recommendations.append("Improve Work-Life Balance by setting clearer boundaries.")
    if data_dict.get('TrainingTimesLastYear', 0) < 2:
        recommendations.append("Increase Training Participation to enhance skills.")
    if data_dict.get('YearsSinceLastPromotion', 0) > 2:
        recommendations.append("Discuss career progression with your manager.")
    if not recommendations:
        recommendations.append("Maintain current performance and seek leadership opportunities.")
    
    company_avg = 70  # Mocked
    diff = prediction - company_avg
    comparison = f"Your score is {abs(diff):.1f}% {'higher' if diff > 0 else 'lower'} than the average {data_dict.get('EmpJobRole', 'Unknown')}."

    return {
        'level': level,
        'color': color,
        'strengths': [f"{k}: {v:.1f}" for k, v in strengths],
        'weaknesses': [f"{k}: {v:.1f}" for k, v in weaknesses],
        'recommendations': recommendations,
        'comparison': comparison
    }

def predict_performance(input_data_raw, model_name):
    input_df = pd.DataFrame([input_data_raw])
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    missing_cols = set(feature_names) - set(input_df_encoded.columns)
    for col in missing_cols:
        input_df_encoded[col] = 0
    extra_cols = set(input_df_encoded.columns) - set(feature_names)
    input_df_encoded = input_df_encoded.drop(columns=extra_cols, errors='ignore')
    input_df_encoded = input_df_encoded[feature_names]
    scaled_input = feature_scaler.transform(input_df_encoded)
    model = model_rf if model_name == 'Random Forest' else model_xg
    return model.predict(scaled_input)[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    charts = {}
    name = None
    insights = None
    selected_model = None
    
    if request.method == 'POST':
        name = request.form['name'].replace(" ", "_")
        selected_model = request.form['model']
        model = model_xg if selected_model == 'XGBoost' else model_rf
        
        input_data_raw = {
            'Age': float(request.form['age']),
            'Gender': request.form['gender'],
            'EducationBackground': request.form['education_background'],
            'MaritalStatus': request.form['marital_status'],
            'EmpDepartment': request.form['department'],
            'EmpJobRole': request.form['job_role'],
            'BusinessTravelFrequency': request.form['travel_frequency'],
            'DistanceFromHome': float(request.form['distance_from_home']),
            'EmpEducationLevel': int(request.form['education_level']),
            'EmpEnvironmentSatisfaction': int(request.form['environment_satisfaction']),
            'EmpHourlyRate': float(request.form['hourly_rate']),
            'EmpJobInvolvement': int(request.form['job_involvement']),
            'EmpJobLevel': int(request.form['job_level']),
            'EmpJobSatisfaction': int(request.form['job_satisfaction']),
            'NumCompaniesWorked': int(request.form['num_companies_worked']),
            'OverTime': request.form['overtime'],
            'EmpLastSalaryHikePercent': float(request.form['last_salary_hike_percent']),
            'EmpRelationshipSatisfaction': int(request.form['relationship_satisfaction']),
            'TotalWorkExperienceInYears': int(request.form['total_work_experience']),
            'TrainingTimesLastYear': int(request.form['training_times_last_year']),
            'EmpWorkLifeBalance': int(request.form['work_life_balance']),
            'ExperienceYearsAtThisCompany': int(request.form['years_at_company']),
            'ExperienceYearsInCurrentRole': int(request.form['years_in_current_role']),
            'YearsSinceLastPromotion': int(request.form['years_since_last_promotion']),
            'YearsWithCurrManager': int(request.form['years_with_curr_manager']),
            'Attrition': request.form['attrition']
        }

        input_df = pd.DataFrame([input_data_raw])
        input_df_encoded = pd.get_dummies(input_df, drop_first=True)
        missing_cols = set(feature_names) - set(input_df_encoded.columns)
        for col in missing_cols:
            input_df_encoded[col] = 0
        extra_cols = set(input_df_encoded.columns) - set(feature_names)
        input_df_encoded = input_df_encoded.drop(columns=extra_cols, errors='ignore')
        input_df_encoded = input_df_encoded[feature_names]
        scaled_input = feature_scaler.transform(input_df_encoded)

        rf_pred = model_rf.predict(scaled_input)[0]
        xg_pred = model_xg.predict(scaled_input)[0]
        prediction = rf_pred if selected_model == 'Random Forest' else xg_pred

        print(f"Random Forest Prediction: {rf_pred}")
        print(f"XGBoost Prediction: {xg_pred}")
        print(f"Selected Model: {selected_model}, Final Prediction: {prediction}")

        insights = get_insights(prediction, input_data_raw)
        charts['bar'] = generate_bar_chart(input_data_raw, name)
        charts['satisfaction'] = generate_satisfaction_bar_chart(input_data_raw, name)
        charts['gauge'] = generate_gauge_chart(prediction, name)
        charts['line'] = generate_line_chart(input_data_raw, name)
        charts['categorical'] = generate_categorical_bar_chart(input_data_raw, name)
        charts['comparison'] = generate_comparison_chart(prediction, name, input_data_raw['EmpJobRole'])

    return render_template('index.html',
                           prediction=prediction,
                           name=name,
                           charts=charts,
                           insights=insights,
                           selected_model=selected_model,
                           genders=GENDERS,
                           edu_backgrounds=EDU_BACKGROUNDS,
                           marital_statuses=MARITAL_STATUSES,
                           departments=DEPARTMENTS,
                           job_roles=JOB_ROLES,
                           travel_frequencies=TRAVEL_FREQUENCIES,
                           overtime_options=OVERTIME_OPTIONS,
                           attrition_options=ATTRITION_OPTIONS,
                           model_options=MODEL_OPTIONS,
                           feature_img='feature_correlation.png',
                           dist_img='performance_distribution.png')

@app.route('/test_ratings', methods=['GET'])
def test_ratings():
    # Test for 90+ rating
    high_performer = {
        'Age': 40.0,
        'Gender': 'Male',
        'EducationBackground': 'Technical Degree',
        'MaritalStatus': 'Married',
        'EmpDepartment': 'Development',
        'EmpJobRole': 'Senior Developer',
        'BusinessTravelFrequency': 'Travel_Rarely',
        'DistanceFromHome': 2.0,
        'EmpEducationLevel': 4,
        'EmpEnvironmentSatisfaction': 4,
        'EmpHourlyRate': 90.0,
        'EmpJobInvolvement': 4,
        'EmpJobLevel': 4,
        'EmpJobSatisfaction': 4,
        'NumCompaniesWorked': 3,
        'OverTime': 'No',
        'EmpLastSalaryHikePercent': 22.0,
        'EmpRelationshipSatisfaction': 4,
        'TotalWorkExperienceInYears': 18,
        'TrainingTimesLastYear': 5,
        'EmpWorkLifeBalance': 4,
        'ExperienceYearsAtThisCompany': 12,
        'ExperienceYearsInCurrentRole': 7,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 6,
        'Attrition': 'No'
    }

    # Test for 20 rating
    low_performer = {
        'Age': 25.0,
        'Gender': 'Female',
        'EducationBackground': 'Other',
        'MaritalStatus': 'Single',
        'EmpDepartment': 'Sales',
        'EmpJobRole': 'Sales Representative',
        'BusinessTravelFrequency': 'Travel_Frequently',
        'DistanceFromHome': 30.0,
        'EmpEducationLevel': 1,
        'EmpEnvironmentSatisfaction': 1,
        'EmpHourlyRate': 40.0,
        'EmpJobInvolvement': 1,
        'EmpJobLevel': 1,
        'EmpJobSatisfaction': 1,
        'NumCompaniesWorked': 8,
        'OverTime': 'Yes',
        'EmpLastSalaryHikePercent': 5.0,
        'EmpRelationshipSatisfaction': 1,
        'TotalWorkExperienceInYears': 2,
        'TrainingTimesLastYear': 0,
        'EmpWorkLifeBalance': 1,
        'ExperienceYearsAtThisCompany': 1,
        'ExperienceYearsInCurrentRole': 0,
        'YearsSinceLastPromotion': 0,
        'YearsWithCurrManager': 0,
        'Attrition': 'Yes'
    }

    rf_90 = predict_performance(high_performer, 'Random Forest')
    xg_90 = predict_performance(high_performer, 'XGBoost')
    rf_20 = predict_performance(low_performer, 'Random Forest')
    xg_20 = predict_performance(low_performer, 'XGBoost')

    return f"""
    <h1>Test Ratings</h1>
    <h2>90+ Rating</h2>
    <p>Random Forest: {rf_90:.2f}</p>
    <p>XGBoost: {xg_90:.2f}</p>
    <h2>20 Rating</h2>
    <p>Random Forest: {rf_20:.2f}</p>
    <p>XGBoost: {xg_20:.2f}</p>
    """

# if __name__ == '__main__':
#     app.run(debug=True)
# ... (previous imports and code remain the same until if __name__ == '__main__')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from env, default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0 for external access
