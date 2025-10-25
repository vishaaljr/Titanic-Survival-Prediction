# Titanic Survival Prediction

Predict whether a passenger survived the Titanic disaster using Machine Learning. This project demonstrates data preprocessing, exploratory data analysis, feature engineering, and model building using classification algorithms.

---

## 🚀 Project Overview

The Titanic dataset is a classic machine learning dataset containing information about passengers, such as age, sex, ticket class, and more. The goal of this project is to predict the survival of passengers based on these features.

**Key Features Used:**

* `Pclass` – Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
* `Sex` – Gender of the passenger
* `Age` – Age of the passenger
* `SibSp` – Number of siblings/spouses aboard
* `Parch` – Number of parents/children aboard
* `Fare` – Ticket fare

---

## 📈 Data Exploration & Preprocessing

1. Handled missing values for columns like `Age` and `Embarked`.
2. Encoded categorical variables (`Sex`, `Embarked`) into numerical form.
3. Scaled continuous features to normalize data.
4. Split dataset into training and testing sets.

---

## 🛠️ Model Building

Classification algorithms were used to predict survival:

* Logistic Regression ✅

The model was evaluated using metrics like:

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📝 How to Use

1. Clone this repository:

```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
```

2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Run the notebook or Python script:

```bash
jupyter notebook Titanic_Survival_Prediction.ipynb
# or
python titanic_prediction.py
```

---

## 🛠️ Example: Predicting Survival for a Single Passenger

You can use your trained model to predict whether an individual passenger survived. Here's a sample Python snippet:

```python
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("models/titanic_model.pkl", "rb"))

# Passenger data
user = [1, 0, 38.0, 1, 0, 71.2833]  # Example: Pclass, Sex, Age, SibSp, Parch, Fare
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Convert to DataFrame
user_df = pd.DataFrame([user], columns=columns)

# Make prediction
prediction = model.predict(user_df)

if prediction[0] == 1:
    print("The person survived.")
else:
    print("The person did not survive.")
```

This will output whether the passenger survived based on the model's prediction.

---

## 📊 Results

The Logistic Regression model achieved an **accuracy of 74%** on the test set.


## 🔧 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook














