import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

def TVT_splitter(x, y, train_ratio, validation_ratio, test_ratio):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    return x_train, x_val, x_test, y_train, y_val, y_test

def dimensionality_reduction(X_trainval, x_test_scaled, n_components):

    pca = PCA(n_components)
    X_trainval_pca = pca.fit_transform(X_trainval)
    X_test_pca = pca.transform(x_test_scaled)

    return X_trainval_pca, X_test_pca

# Carregar dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist["data"], mnist["target"].astype(np.uint8) 

# Dividir em treino, validação e teste
x_train, x_val, x_test, y_train, y_val, y_test = TVT_splitter(x, y, train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO, test_ratio=TEST_RATIO)

# Padronizar
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Grids de hiperparâmetros
n_components_grid = [20, 40, 60, 80, 100, 200, x.shape[1]/2]
n_neighbors_grid = [3, 5, 7, 9]
max_iter_grid = [500, 1000, 2000]
metrics_grid = ['euclidean', 'manhattan']

# 🔥 Cache de PCA para não recalcular
pca_cache = {}
for n_components in n_components_grid:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(x_train_scaled)
    X_val_pca = pca.transform(x_val_scaled)
    pca_cache[n_components] = (X_train_pca, X_val_pca, pca)

# Otimização dos hiperparâmetros do kNN usando os dados de validação
best_knn_acc = 0
best_knn_params = {}
print("choosing hyperparameters for KNN...")
for n_components in n_components_grid:
    X_train_pca, X_val_pca, _ = pca_cache[n_components]
    for n_neighbors in n_neighbors_grid:
        for metric in metrics_grid:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
            knn.fit(X_train_pca, y_train)
            y_pred_val = knn.predict(X_val_pca)
            acc = accuracy_score(y_val, y_pred_val)
            if acc > best_knn_acc:
                best_knn_acc = acc
                best_knn_params = {
                    'n_components': n_components,
                    'n_neighbors': n_neighbors,
                    'metric': metric
                }

print(f"Best kNN params: {best_knn_params}, Validation Accuracy: {best_knn_acc:.4f}")

# Otimização dos hiperparâmetros da Regressão Logística usando os dados de validação
best_log_acc = 0
best_log_params = {}
print("choosing hyperparameters for Logistic Regression...")
for n_components in n_components_grid:
    X_train_pca, X_val_pca, _ = pca_cache[n_components]
    for max_iter in max_iter_grid:
        clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
        clf.fit(X_train_pca, y_train)
        y_pred_val = clf.predict(X_val_pca)
        acc = accuracy_score(y_val, y_pred_val)
        if acc > best_log_acc:
            best_log_acc = acc
            best_log_params = {
                'n_components': n_components,
                'max_iter': max_iter
            }

print(f"Best Logistic Regression params: {best_log_params}, Validation Accuracy: {best_log_acc:.4f}")

# Otimização dos hiperparâmetros Naive Bayes usando dos dados de validação
best_nb_acc = 0
best_nb_params = {}
print("choosing hyperparameters for Naive Bayes...")
for n_components in n_components_grid:
    X_train_pca, X_val_pca, _ = pca_cache[n_components]
    nb = GaussianNB()
    nb.fit(X_train_pca, y_train)
    y_pred_val = nb.predict(X_val_pca)
    acc = accuracy_score(y_val, y_pred_val)
    if acc > best_nb_acc:
        best_nb_acc = acc
        best_nb_params = {
            'n_components': n_components
        }

print(f"Best Naive Bayes params: {best_nb_params}, Validation Accuracy: {best_nb_acc:.4f}")

# Unir treino + validação para teste final
X_trainval = np.vstack([x_train_scaled, x_val_scaled])
y_trainval = np.concatenate([y_train, y_val])

# kNN
X_trainval_pca, X_test_pca = dimensionality_reduction(X_trainval, x_test_scaled, best_knn_params['n_components'])
knn = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], metric=best_knn_params['metric'], n_jobs=-1)
knn.fit(X_trainval_pca, y_trainval)
y_pred_knn = knn.predict(X_test_pca)
print("Test Accuracy kNN:", accuracy_score(y_test, y_pred_knn))

# Logistic Regression
X_trainval_pca, X_test_pca = dimensionality_reduction(X_trainval, x_test_scaled, best_log_params['n_components'])
clf = LogisticRegression(max_iter=best_log_params['max_iter'], n_jobs=-1)
clf.fit(X_trainval_pca, y_trainval)
y_pred_log = clf.predict(X_test_pca)
print("Test Accuracy Logistic Regression:", accuracy_score(y_test, y_pred_log))

# Naive Bayes
X_trainval_pca, X_test_pca, _ = dimensionality_reduction(X_trainval, x_test_scaled, best_nb_params['n_components'])
nb = GaussianNB()
nb.fit(X_trainval_pca, y_trainval)
y_pred_nb = nb.predict(X_test_pca)
print("Test Accuracy Naive Bayes:", accuracy_score(y_test, y_pred_nb))
