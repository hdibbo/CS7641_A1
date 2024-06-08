import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Generate synthetic data
np.random.seed(42)
num_samples = 1000
age = np.random.randint(18, 80, size=num_samples)
gender = np.random.choice(['M', 'F'], size=num_samples)
weight = np.round(np.random.uniform(50, 100, size=num_samples), 1)
height = np.round(np.random.uniform(150, 200, size=num_samples), 1)
medical_history = np.random.choice(['Diabetes', 'Hypertension', 'None'], size=num_samples)
current_medications = np.random.choice(['DrugA', 'DrugB', 'DrugC', 'DrugD', 'DrugE', 'DrugF'], size=(num_samples, 2))
dosage = np.round(np.random.uniform(50, 300, size=num_samples), 1)
duration_of_treatment = np.random.randint(10, 100, size=num_samples)
adr_occurrence = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Weight': weight,
    'Height': height,
    'Medical History': medical_history,
    'Current Medications': [list(meds) for meds in current_medications],
    'Dosage': dosage,
    'Duration of Treatment': duration_of_treatment,
    'ADR Occurrence': adr_occurrence
})

# Preprocess data
# Expand the current medications into separate columns
for med in ['DrugA', 'DrugB', 'DrugC', 'DrugD', 'DrugE', 'DrugF']:
    data[med] = data['Current Medications'].apply(lambda meds: med in meds)

# Drop the original 'Current Medications' column
data = data.drop('Current Medications', axis=1)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['Gender', 'Medical History'])

# Separate features and target
X = data.drop('ADR Occurrence', axis=1)
y = data['ADR Occurrence']

# Check class distribution
print("Class distribution before resampling:")
print(y.value_counts())

# Resample data to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after resampling
print("Class distribution after resampling:")
print(y_resampled.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Neural Network
nn = MLPClassifier(max_iter=500, random_state=42)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print("Neural Network Results")
print(classification_report(y_test, y_pred_nn))

# SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', random_state=42)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid_svm_rbf = GridSearchCV(svm_rbf, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = grid_svm_rbf.best_estimator_.predict(X_test)
print("SVM with RBF Kernel Results")
print(classification_report(y_test, y_pred_svm_rbf))
print(f"Best parameters for SVM with RBF: {grid_svm_rbf.best_params_}")

# SVM with Linear Kernel
svm_linear = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
print("SVM with Linear Kernel Results")
print(classification_report(y_test, y_pred_svm_linear))

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("k-Nearest Neighbors Results")
print(classification_report(y_test, y_pred_knn))

# Plot Learning Curves
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Error rate")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_errors = 1 - np.mean(train_scores, axis=1)
    test_errors = 1 - np.mean(test_scores, axis=1)
    train_errors_std = np.std(1 - train_scores, axis=1)
    test_errors_std = np.std(1 - test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_errors - train_errors_std,
                     train_errors + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors - test_errors_std,
                     test_errors + test_errors_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_errors, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_errors, 'o-', color="g",
             label="Cross-validation error")
    
    plt.legend(loc="best")
    return plt

plot_learning_curve(nn, "Learning Curves (Neural Network)", X, y, cv=5)
plot_learning_curve(grid_svm_rbf.best_estimator_, "Learning Curves (SVM with RBF Kernel)", X, y, cv=5)
plot_learning_curve(svm_linear, "Learning Curves (SVM with Linear Kernel)", X, y, cv=5)
plot_learning_curve(knn, "Learning Curves (k-Nearest Neighbors)", X, y, cv=5)
plt.show()

# Plot Validation Curves
def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=None, n_jobs=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Error rate")
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=n_jobs)
    train_errors = 1 - np.mean(train_scores, axis=1)
    test_errors = 1 - np.mean(test_scores, axis=1)
    train_errors_std = np.std(1 - train_scores, axis=1)
    test_errors_std = np.std(1 - test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(param_range, train_errors - train_errors_std,
                     train_errors + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_range, test_errors - test_errors_std,
                     test_errors + test_errors_std, alpha=0.1, color="g")
    plt.plot(param_range, train_errors, 'o-', color="r",
             label="Training error")
    plt.plot(param_range, test_errors, 'o-', color="g",
             label="Cross-validation error")
    
    plt.legend(loc="best")
    return plt

# Neural Network Validation Curve
plot_validation_curve(nn, "Validation Curve (Neural Network)", X, y, param_name='alpha', param_range=np.logspace(-6, -1, 6), cv=5)
# SVM with RBF Kernel Validation Curve
plot_validation_curve(grid_svm_rbf.best_estimator_, "Validation Curve (SVM with RBF Kernel)", X, y, param_name='C', param_range=[0.1, 1, 10, 100], cv=5)
# SVM with Linear Kernel Validation Curve
plot_validation_curve(svm_linear, "Validation Curve (SVM with Linear Kernel)", X, y, param_name='C', param_range=[0.1, 1, 10, 100], cv=5)
# k-Nearest Neighbors Validation Curve
plot_validation_curve(knn, "Validation Curve (k-Nearest Neighbors)", X, y, param_name='n_neighbors', param_range=np.arange(1, 31), cv=5)
plt.show()
