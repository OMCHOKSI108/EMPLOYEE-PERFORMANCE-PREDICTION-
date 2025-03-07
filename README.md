 
# Employee Performance Prediction

This project predicts employee performance ratings using machine learning models (Random Forest and XGBoost) based on various employee attributes. The application provides a Flask-based web interface to input employee details and visualize performance predictions with charts and insights.


[Employee Performance Prediction App](https://employee-performance-prediction-production.up.railway.app/)

## Project Structure
```
EMPLOYEE-PERFORMANCE-PREDICTION-/
├── train_data.py          # Script to train and save models
├── app.py                # Flask app for predictions and visualization
├── backend/              # Directory for trained models and artifacts
│   ├── model_RF.pkl
│   ├── model_xg.pkl
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   ├── mappings.pkl
│   └── feature_names.pkl
├── static/               # Directory for static files and charts
│   ├── charts/          # Generated charts (created at runtime)
│   ├── feature_correlation.png
│   └── performance_distribution.png
├── templates/            # Directory for HTML templates
│   └── index.html
├── INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv  # Dataset
└── README.md             # This file
```
## Project Structure
```
EMPLOYEE-PERFORMANCE-PREDICTION-/
├── train_data.py          # Script to train and save models
├── app.py                # Flask app for predictions and visualization
├── backend/              # Directory for trained models and artifacts
│   ├── model_RF.pkl
│   ├── model_xg.pkl
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   ├── mappings.pkl
│   └── feature_names.pkl
├── static/               # Directory for static files and charts
│   ├── charts/          # Generated charts (created at runtime)
│   ├── feature_correlation.png
│   └── performance_distribution.png
├── templates/            # Directory for HTML templates
│   └── index.html
├── INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv  # Dataset
└── README.md             # This file
```

## Prerequisites
- Python 3.8+
- Required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `flask`
  - `matplotlib`

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost flask matplotlib
```

## Dataset
- **File**: `INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv`
- Place this CSV file in the project root directory (`D:\PROJECTS\EPP_PROJ\EMPLOYEE-PERFORMANCE-PREDICTION-`).
- The dataset contains employee attributes (e.g., `Age`, `Gender`, `EmpJobSatisfaction`) and a target column `PerformanceRating` (originally 1-4, scaled to 1-100).

## Training Models
The `train_data.py` script trains two models:
- **Random Forest Regressor**: Predicts performance with increased flexibility (`max_depth=15`).
- **XGBoost Regressor**: Boosted tree model for robust predictions.

### Steps to Train
1. Ensure the dataset is in the root directory.
2. (Optional) Delete the old `backend/` folder to retrain:
   ```bash
   rd /s /q D:\PROJECTS\EPP_PROJ\EMPLOYEE-PERFORMANCE-PREDICTION-\backend
   ```
3. Run the script:
   ```bash
   cd D:\PROJECTS\EPP_PROJ\EMPLOYEE-PERFORMANCE-PREDICTION-
   python train_data.py
   ```
4. Output:
   - Model performance metrics (R², MAE).
   - First 5 predictions for both models.
   - Feature importance for Random Forest and XGBoost.
   - Saved files in `backend/`:
     - `model_RF.pkl`: Random Forest model
     - `model_xg.pkl`: XGBoost model
     - `feature_scaler.pkl`: Feature scaler
     - `target_scaler.pkl`: Target scaler
     - `mappings.pkl`: Categorical mappings
     - `feature_names.pkl`: Feature names after one-hot encoding

## Running the Flask App
The `app.py` script provides a web interface to:
- Input employee details.
- Predict performance using either Random Forest or XGBoost.
- Visualize results with charts (bar, gauge, line, etc.).
- Test specific ratings (90+ and 20) via `/test_ratings`.

### Steps to Run
1. Ensure all `backend/` files are present from training.
2. Create `templates/index.html` (see below if not already created).
3. Add placeholder images to `static/`:
   - `feature_correlation.png`
   - `performance_distribution.png`
   - Or update `app.py` to remove these from `render_template` if not needed.
4. Run the app:
   ```bash
   cd D:\PROJECTS\EPP_PROJ\EMPLOYEE-PERFORMANCE-PREDICTION-
   python app.py
   ```
5. Access the app:
   - Main interface: `http://127.0.0.1:5000/`
   - Test ratings: `http://127.0.0.1:5000/test_ratings`

### Features
- **Form Input**: Enter employee details (e.g., Age, Job Satisfaction, Department).
- **Model Selection**: Choose between Random Forest and XGBoost.
- **Charts**: Visualize key metrics, satisfaction, performance score, and more.
- **Insights**: Performance level (Low, Moderate, High), strengths, weaknesses, and recommendations.
- **Test Ratings**: Check example inputs for 90+ and 20 ratings.

## Example `index.html`
If you don’t have `templates/index.html`, here’s a minimal version to get started:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Employee Performance Prediction</title>
</head>
<body>
    <h1>Employee Performance Prediction</h1>
    <form method="POST">
        <label>Name: <input type="text" name="name" required></label><br>
        <label>Age: <input type="number" name="age" required></label><br>
        <label>Gender: <select name="gender">
            {% for gender in genders %}<option value="{{ gender }}">{{ gender }}</option>{% endfor %}
        </select></label><br>
        <label>Education Background: <select name="education_background">
            {% for edu in edu_backgrounds %}<option value="{{ edu }}">{{ edu }}</option>{% endfor %}
        </select></label><br>
        <label>Marital Status: <select name="marital_status">
            {% for status in marital_statuses %}<option value="{{ status }}">{{ status }}</option>{% endfor %}
        </select></label><br>
        <label>Department: <select name="department">
            {% for dept in departments %}<option value="{{ dept }}">{{ dept }}</option>{% endfor %}
        </select></label><br>
        <label>Job Role: <select name="job_role">
            {% for role in job_roles %}<option value="{{ role }}">{{ role }}</option>{% endfor %}
        </select></label><br>
        <label>Travel Frequency: <select name="travel_frequency">
            {% for freq in travel_frequencies %}<option value="{{ freq }}">{{ freq }}</option>{% endfor %}
        </select></label><br>
        <label>Distance From Home: <input type="number" name="distance_from_home" required></label><br>
        <label>Education Level: <input type="number" name="education_level" min="1" max="5" required></label><br>
        <label>Environment Satisfaction: <input type="number" name="environment_satisfaction" min="1" max="4" required></label><br>
        <label>Hourly Rate: <input type="number" name="hourly_rate" required></label><br>
        <label>Job Involvement: <input type="number" name="job_involvement" min="1" max="4" required></label><br>
        <label>Job Level: <input type="number" name="job_level" min="1" max="5" required></label><br>
        <label>Job Satisfaction: <input type="number" name="job_satisfaction" min="1" max="4" required></label><br>
        <label>Num Companies Worked: <input type="number" name="num_companies_worked" required></label><br>
        <label>OverTime: <select name="overtime">
            {% for ot in overtime_options %}<option value="{{ ot }}">{{ ot }}</option>{% endfor %}
        </select></label><br>
        <label>Last Salary Hike (%): <input type="number" name="last_salary_hike_percent" required></label><br>
        <label>Relationship Satisfaction: <input type="number" name="relationship_satisfaction" min="1" max="4" required></label><br>
        <label>Total Work Experience: <input type="number" name="total_work_experience" required></label><br>
        <label>Training Times Last Year: <input type="number" name="training_times_last_year" required></label><br>
        <label>Work Life Balance: <input type="number" name="work_life_balance" min="1" max="4" required></label><br>
        <label>Years at Company: <input type="number" name="years_at_company" required></label><br>
        <label>Years in Current Role: <input type="number" name="years_in_current_role" required></label><br>
        <label>Years Since Last Promotion: <input type="number" name="years_since_last_promotion" required></label><br>
        <label>Years With Current Manager: <input type="number" name="years_with_curr_manager" required></label><br>
        <label>Attrition: <select name="attrition">
            {% for attr in attrition_options %}<option value="{{ attr }}">{{ attr }}</option>{% endfor %}
        </select></label><br>
        <label>Model: <select name="model">
            {% for model in model_options %}<option value="{{ model }}">{{ model }}</option>{% endfor %}
        </select></label><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
    <h2>Prediction for {{ name }}</h2>
    <p>Model: {{ selected_model }}</p>
    <p>Performance Score: {{ prediction|round(2) }}</p>
    <h3>Insights</h3>
    <p>Level: <span style="color: {{ insights.color }}">{{ insights.level }}</span></p>
    <p>Strengths: {{ insights.strengths|join(', ') }}</p>
    <p>Weaknesses: {{ insights.weaknesses|join(', ') }}</p>
    <p>Recommendations: {{ insights.recommendations|join(', ') }}</p>
    <p>{{ insights.comparison }}</p>
    <h3>Charts</h3>
    <img src="{{ url_for('static', filename=charts.bar) }}" alt="Bar Chart">
    <img src="{{ url_for('static', filename=charts.satisfaction) }}" alt="Satisfaction Chart">
    <img src="{{ url_for('static', filename=charts.gauge) }}" alt="Gauge Chart">
    <img src="{{ url_for('static', filename=charts.line) }}" alt="Line Chart">
    <img src="{{ url_for('static', filename=charts.categorical) }}" alt="Categorical Chart">
    <img src="{{ url_for('static', filename=charts.comparison) }}" alt="Comparison Chart">
    {% endif %}
</body>
</html>
```
- Save this as `templates/index.html`.

## Model-Based Criteria for Ratings
### 90+ Rating
- **Profile**: High-performing, experienced employee.
- **Choices**:
  - `EmpEnvironmentSatisfaction`: 4
  - `EmpJobSatisfaction`: 4
  - `EmpWorkLifeBalance`: 4
  - `EmpRelationshipSatisfaction`: 4
  - `EmpLastSalaryHikePercent`: 22%
  - `EmpJobInvolvement`: 4
  - `EmpJobLevel`: 4
  - `TotalWorkExperienceInYears`: 18
  - `TrainingTimesLastYear`: 5
  - `ExperienceYearsAtThisCompany`: 12
  - `Gender`: 'Male'
  - `EmpDepartment`: 'Development'
  - `EmpJobRole`: 'Senior Developer'
  - `OverTime`: 'No'
  - `Attrition`: 'No'
- **Test**: Visit `/test_ratings` to verify.

### 20 Rating
- **Profile**: Low-performing, disengaged employee.
- **Choices**:
  - `EmpEnvironmentSatisfaction`: 1
  - `EmpJobSatisfaction`: 1
  - `EmpWorkLifeBalance`: 1
  - `EmpRelationshipSatisfaction`: 1
  - `EmpLastSalaryHikePercent`: 5%
  - `EmpJobInvolvement`: 1
  - `EmpJobLevel`: 1
  - `TotalWorkExperienceInYears`: 2
  - `TrainingTimesLastYear`: 0
  - `ExperienceYearsAtThisCompany`: 1
  - `Gender`: 'Female'
  - `EmpDepartment`: 'Sales'
  - `EmpJobRole`: 'Sales Representative'
  - `OverTime`: 'Yes'
  - `Attrition`: 'Yes'
- **Test**: Visit `/test_ratings` to verify.

## Troubleshooting
- **FileNotFoundError**: Ensure all `backend/` files exist after training.
- **Constant Predictions**: If Random Forest predicts `3`, retrain with `train_data.py` after deleting `backend/model_RF.pkl`.
- **Narrow Range (50-60)**: Increase `max_depth` further in `train_data.py` (e.g., `None`) and retrain.
- **Charts Not Showing**: Verify `static/charts/` is writable and `index.html` matches `charts` keys.

## Contributing
Feel free to fork this repository, enhance the models, or improve the UI. Submit pull requests with your changes.

## License
This project is unlicensed—use it freely!

---

 
