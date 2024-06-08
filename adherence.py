import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
df = pd.read_csv('/Users/david_h/Downloads/medication_adherence.csv')

# Display basic information about the dataset
print(df.head())
print(df.info())

# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split the data into features and target variable
X = df.drop('adherence', axis=1)
y = df['adherence']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Networks
nn_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='lbfgs', max_iter=2000, random_state=42)

# Learning curve for Neural Network
train_sizes, train_scores, valid_scores = learning_curve(nn_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, valid_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve for Neural Network')
plt.legend()

# Validation curve for Neural Network
param_range = np.logspace(-4, 1, 6)
train_scores, valid_scores = validation_curve(nn_model, X_train, y_train, param_name='alpha', param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.subplot(2, 2, 2)
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, valid_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Parameter alpha')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Validation Curve for Neural Network')
plt.legend()

plt.tight_layout()
plt.show()

# SVM with Linear Kernel
svm_linear = SVC(kernel='linear', probability=True, random_state=42)

# Learning curve for SVM (Linear Kernel)
train_sizes, train_scores, valid_scores = learning_curve(svm_linear, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, valid_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve for SVM (Linear Kernel)')
plt.legend()

# Validation curve for SVM (Linear Kernel)
param_range = np.logspace(-4, 1, 6)
train_scores, valid_scores = validation_curve(svm_linear, X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.subplot(2, 2, 2)
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, valid_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Validation Curve for SVM (Linear Kernel)')
plt.legend()

plt.tight_layout()
plt.show()

# SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

# Learning curve for SVM (RBF Kernel)
train_sizes, train_scores, valid_scores = learning_curve(svm_rbf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, valid_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve for SVM (RBF Kernel)')
plt.legend()

# Validation curve for SVM (RBF Kernel)
param_range = np.logspace(-4, 1, 6)
train_scores, valid_scores = validation_curve(svm_rbf, X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.subplot(2, 2, 2)
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, valid_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Validation Curve for SVM (RBF Kernel)')
plt.legend()

plt.tight_layout()
plt.show()

# k-Nearest Neighbors (k-NN)
k_values = [3, 5, 7, 11]
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)

# Learning curve for k-NN
train_sizes, train_scores, valid_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, valid_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title(f'Learning Curve for k-NN (k={best_k})')
plt.legend()

# Validation curve for k-NN
param_range = np.arange(1, 21)
train_scores, valid_scores = validation_curve(knn, X_train, y_train, param_name='n_neighbors', param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.subplot(2, 2, 2)
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, valid_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title(f'Validation Curve for k-NN')
plt.legend()

plt.tight_layout()
plt.show()


