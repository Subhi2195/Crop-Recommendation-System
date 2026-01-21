# train.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from statistics import mean,stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("data/Crop_recommendation.csv")

# Encode label
# le = LabelEncoder()
# df['label'] = le.fit_transform(df['label'])

# Features and target
X = df.drop('label', axis=1)
y = df['label']


# 1. THE SPLIT: Immediate isolation of the 'Vault'
# We use 20% (400 rows) as the absolute final test
X_dev, X_vault, y_dev, y_vault = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 2. CROSS-VALIDATION (On Dev Set Only)
# This mimics training on different "slices" of your known data
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(pipeline, X_dev, y_dev, cv=skf)

print(f"Development CV Mean Accuracy: {cv_scores.mean():.4f}")

# 3. THE FINAL REHEARSAL
# Now we train on the FULL 80% and test on the 20% 'Vault'
pipeline.fit(X_dev, y_dev)
vault_predictions = pipeline.predict(X_vault)

# 4. ANALYSIS
print("--- Final Vault Results ---")
print(classification_report(y_vault, vault_predictions))

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)

# # Train model
# model = RandomForestClassifier(random_state=42)

# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)
# lst_accu_stratified = []

# for train_index, test_index in skf.split(X,y):
#     x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index]
#     y_train_fold, y_test_fold = y[train_index], y[test_index]
#     model.fit(x_train_fold, y_train_fold)
#     lst_accu_stratified.append(model.score(x_test_fold,y_test_fold))


# #To check if there was any data leakage on performing it before CV- but no change
# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train)
# y_test_encoded = le.transform(y_test)
# model.fit(X_train, y_train)

# # Evaluate
# print('List of possible accuracy: ', lst_accu_stratified)
# print(f"\nMaximum Accuracy That can be obtained from model is:", max(lst_accu_stratified)*100, '%')
# print('\nMinimum Accuracy:',
# 	min(lst_accu_stratified)*100, '%')
# print('\nOverall Accuracy:',
# 	mean(lst_accu_stratified)*100, '%')
# print('\nStandard Deviation is:', stdev(lst_accu_stratified))

# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# # Save model and label encoder
# joblib.dump(model, "crop_model.pkl")
# joblib.dump(le, "label_encoder.pkl")

# print("Model and label encoder saved.")

# # Feature importance
# importances = model.feature_importances_
# feature_names = X.columns

# # Create a DataFrame for plotting
# feat_imp_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
# plt.title('Feature Importance (Random Forest)')
# plt.tight_layout()
# plt.savefig('feature_importance.png')  # Save the plot
# plt.show()
