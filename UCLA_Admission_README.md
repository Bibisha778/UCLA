# 🎓 UCLA Admission Chance Predictor

Estimate a student’s chance of admission to UCLA using GRE, TOEFL, CGPA, and research experience via a neural network.

## 🚀 Features
- Predicts admission probability (0.0 - 1.0)
- Trained using `MLPRegressor` (Neural Network)
- Streamlit web UI for real-time prediction
- Residual plot for evaluation
- RMSE and R² score included

## 📂 Project Structure
- `streamlit.py`: Streamlit app
- `main.py`: CLI trainer
- `models/`: Trained neural network
- `src/`: Data loader, scaler, model training, and visualization

## ▶️ Run Locally
```bash
streamlit run streamlit.py
```

## 🧰 Requirements
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn