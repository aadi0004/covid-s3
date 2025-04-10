import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("covid_toy.csv")
df = df.dropna()

# Label Encoding
lb = LabelEncoder()
df['gender'] = lb.fit_transform(df['gender'])
df['city'] = lb.fit_transform(df['city'])
df['fever'] = lb.fit_transform(df['fever'])
df['cough'] = lb.fit_transform(df['cough'])
df['has_covid'] = lb.fit_transform(df['has_covid'])

# Features & Target
X = df[['age', 'gender', 'fever', 'cough', 'city']]
y = df['has_covid']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scale age
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Model & GridSearch
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save best model only
best_model = grid_search.best_estimator_
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved as model.pkl ✅")
