# Diabetes Risk Prediction App

![App Screenshot](https://via.placeholder.com/800x450.png?text=Diabetes+Risk+Prediction+App)
*A web-based ML application for assessing diabetes risk*

## Features

- 🩺 Diabetes risk prediction using machine learning
- 📊 Interactive form for health metrics input
- 🧪 Sample data buttons for quick demonstrations
- 🔍 Real-time risk probability calculation
- 📈 Automatic data imputation for missing values
- 🚦 Clear visual indicators for high/low risk
- 📱 Mobile-responsive design

## Prerequisites

- Python 3.7+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction-app.git
cd diabetes-prediction-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```txt
Flask==2.0.1
scikit-learn==0.24.2
pandas==1.2.4
numpy==1.20.3
joblib==1.0.1
gunicorn==20.1.0
python-dotenv==0.19.0
Werkzeug==2.0.3
```

## Configuration

Create a `.env` file:
```bash
SECRET_KEY=your_secret_key_here
```

## Usage

Run the development server:
```bash
flask run
```

For production:
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

Access the app at: `http://localhost:5000`

## API Endpoints

`POST /predict` - Receives JSON input and returns prediction:
```json
{
  "pregnancies": 2,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50
}
```

## Data Description

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-17 |
| Glucose | Plasma glucose concentration | 0-200 mg/dL |
| Blood Pressure | Diastolic blood pressure | 0-122 mm Hg |
| Skin Thickness | Triceps skinfold thickness | 0-99 mm |
| Insulin | 2-Hour serum insulin | 0-846 μU/ml |
| BMI | Body mass index | 0-67.1 kg/m² |
| Diabetes Pedigree | Diabetes likelihood score | 0.08-2.42 |
| Age | Age in years | 21-81 |

## License

MIT License

## References

- Pima Indians Diabetes Dataset (Kaggle)
- Scikit-learn Documentation
- Flask Documentation
