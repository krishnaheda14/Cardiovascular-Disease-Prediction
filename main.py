import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

# Step 1: Load and Explore the Dataset
data = pd.read_csv('cardiovascular disease data.csv')

# Step 3: Data Preprocessing
# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['target'])
y = data['target']

# Step 4: Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Choose Machine Learning Models (Random Forest, Logistic Regression, Naive Bayes, SVM, KNN, Gradient Boosting, XGBoost)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(max_iter=1000)
nb_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5)
xgb_model = xgb.XGBClassifier(random_state=42)

# Step 7: Train and Evaluate Models with 10-Fold Cross-Validation
models = [rf_model, lr_model, nb_model,  knn_model,  xgb_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'KNN', 'XGBoost']

for model, model_name in zip(models, model_names):
    # Perform 10-fold cross-validation and calculate the mean accuracy
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    mean_accuracy = cv_scores.mean()
    print(f"{model_name} Model (10-Fold Cross-Validation):")
    print("Mean Accuracy:", mean_accuracy)
    print()

# Step 8: Combine models using VotingClassifier
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lr', lr_model),
        ('nb', nb_model),
        ('knn', knn_model),
        ('xgb', xgb_model)
    ],
    voting='hard'  # Use 'hard' for majority voting
)

voting_classifier.fit(X_train, y_train)
voting_y_pred = voting_classifier.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_y_pred)
voting_report = classification_report(y_test, voting_y_pred)

# Print the accuracy of the Voting Classifier
print("Voting Classifier (Ensemble) Model:")
print("Accuracy:", voting_accuracy)
print(voting_report)
