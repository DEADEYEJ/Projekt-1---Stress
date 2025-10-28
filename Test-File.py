import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('Stress.csv')

data.drop(["Course", "Gender", "Depression_Score", "Anxiety_Score", "Relationship_Status", "Substance_Use", "Extracurricular_Involvement", "Residence_Type"],axis=1,inplace=True) # Drop columns we don't need
data.dropna(inplace=True) # Drop nan rows
data['Sleep_Quality'].replace(['Good','Average','Poor'],[2,1,0],inplace=True) # Replace strings with 0 and 1
data['Diet_Quality'].replace(['Good','Average','Poor'],[2,1,0],inplace=True)
data['Physical_Activity'].replace(['High','Moderate','Low'],[2,1,0],inplace=True)
data['Social_Support'].replace(["High", "Moderate", "Low"],[2,1,0],inplace=True)
data["Counseling_Service_Use"].replace(["Frequently", "Occasionally", "Never"],[2,1,0],inplace=True)
data["Family_History"].replace(["Yes","No"],[1,0],inplace=True)
data["Chronic_Illness"].replace(["Yes","No"],[1,0],inplace=True)
data["Financial_Stress"].replace(["High","Moderate","Low"],[2,1,0],inplace=True)

# Kept Data : Age, CGPA, Stress_Level, Sleep_Quality, Physical_Activity, Diet_Quality, Social_Support, Counseling_Service_Use, Family_History, Chronic_Illness, Financial_Stress, Semester_Credit_Load

# Define data
X = data[["Sleep_Quality", "Diet_Quality", "Physical_Activity", "Social_Support", "Counseling_Service_Use", "Family_History", "Chronic_Illness", "Semester_Credit_Load"]].values
y = data[["Stress_Level"]].values

# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_validation, X_test_validation, y_train_validation, y_test_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42) # validation set


# Define classifiers
knn_classifier = KNeighborsClassifier(n_neighbors=20) # Change hyperparameters
decision_tree_classifier = DecisionTreeClassifier(max_depth=3,min_samples_leaf=10, random_state=42) # Change hyperparameters
random_forest_classifier = RandomForestClassifier(n_estimators=500,bootstrap=True,min_samples_leaf=10, random_state=42) # Change hyperparameters

# Train classifiers
knn_classifier.fit(X_train, y_train)
decision_tree_classifier.fit(X_train, y_train)
random_forest_classifier.fit(X_train, y_train)

# Make predictions on both training and testing sets
y_pred_knn_train = knn_classifier.predict(X_train)
y_pred_knn_test = knn_classifier.predict(X_test_validation)

y_pred_tree_train = decision_tree_classifier.predict(X_train)
y_pred_tree_test = decision_tree_classifier.predict(X_test_validation)

y_pred_rf_train = random_forest_classifier.predict(X_train)
y_pred_rf_test = random_forest_classifier.predict(X_test_validation)

# Calculate training and testing accuracy
accuracy_knn_train = accuracy_score(y_train, y_pred_knn_train)
accuracy_knn_test = accuracy_score(y_test_validation, y_pred_knn_test)

accuracy_tree_train = accuracy_score(y_train, y_pred_tree_train)
accuracy_tree_test = accuracy_score(y_test_validation, y_pred_tree_test)

accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
accuracy_rf_test = accuracy_score(y_test_validation, y_pred_rf_test)

# Print the results
print('Model accuracy')
print('\nkNN')
print(f' - Training: {accuracy_knn_train * 100:.2f}%')
print(f' - Test: {accuracy_knn_test * 100:.2f}%')
print('\nDecision tree')
print(f' - Training: {accuracy_tree_train * 100:.2f}%')
print(f' - Test: {accuracy_tree_test * 100:.2f}%')
print('\nRandom forest')
print(f' - Training: {accuracy_rf_train * 100:.2f}%')
print(f' - Test: {accuracy_rf_test * 100:.2f}%')




















