import os
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
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

def TVT_splitter(x, y, train_ratio, validation_ratio, test_ratio):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    return x_train, x_val, x_test, y_train, y_val, y_test

def dimensionality_reduction(x_train, x_test_scaled, n_components):

    pca = PCA(n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test_scaled)

    return x_train_pca, x_test_pca

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
n_components_grid = [20, 40, 60, 80, 100]
n_neighbors_grid = [3, 5, 7, 9]
max_iter_grid = [500, 1000, 2000]

# 🔥 Cache de PCA para não recalcular
pca_cache = {}
for n_components in n_components_grid:
    pca_cache[n_components] = dimensionality_reduction(x_train_scaled, x_val_scaled, n_components)

# Otimização dos hiperparâmetros do kNN com distancia euclidiana usando os dados de validação
best_knn_euc_acc = 0
best_knn_euc_params = {}
print("Choosing hyperparameters for KNN Euclidean...")
for i, n_components in enumerate(n_components_grid):
    print(f"Component: {i+1}/{len(n_components_grid)}, n_components: {n_components}")
    X_train_pca, X_val_pca = pca_cache[n_components]
    for j, n_neighbors in enumerate(n_neighbors_grid):
        print(f"  Neighbors: {j+1}/{len(n_neighbors_grid)}, n_neighbors: {n_neighbors}")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1)
        knn.fit(X_train_pca, y_train)
        y_pred_val = knn.predict(X_val_pca)
        acc = accuracy_score(y_val, y_pred_val)
        if acc > best_knn_euc_acc:
            best_knn_euc_acc = acc
            best_knn_euc_params = {
                'n_components': n_components,
                'n_neighbors': n_neighbors,
            }

print(f"Best kNN euclidian params: {best_knn_euc_params}, Validation Accuracy: {best_knn_euc_acc:.4f}")

# Otimização dos hiperparâmetros do kNN com distancia de manhattan usando os dados de validação
best_knn_man_acc = 0
best_knn_man_params = {}
print("Choosing hyperparameters for KNN Manhattan...")
for i, n_components in enumerate(n_components_grid):
    print(f"Component: {i+1}/{len(n_components_grid)}, n_components: {n_components}")
    X_train_pca, X_val_pca = pca_cache[n_components]
    for j, n_neighbors in enumerate(n_neighbors_grid):
        print(f"  Neighbors: {j+1}/{len(n_neighbors_grid)}, n_neighbors: {n_neighbors}")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan', n_jobs=-1)
        knn.fit(X_train_pca, y_train)
        y_pred_val = knn.predict(X_val_pca)
        acc = accuracy_score(y_val, y_pred_val)
        if acc > best_knn_man_acc:
            best_knn_man_acc = acc
            best_knn_man_params = {
                'n_components': n_components,
                'n_neighbors': n_neighbors,
            }

print(f"Best kNN manhattan params: {best_knn_man_params}, Validation Accuracy: {best_knn_man_acc:.4f}")

# Otimização dos hiperparâmetros da Regressão Logística usando os dados de validação
best_log_acc = 0
best_log_params = {}
print("Choosing hyperparameters for Logistic Regression...")
for i, n_components in enumerate(n_components_grid):
    print(f"Component: {i+1}/{len(n_components_grid)}, n_components: {n_components}")
    X_train_pca, X_val_pca = pca_cache[n_components]
    for j, max_iter in enumerate(max_iter_grid):
        print(f"  Iterations: {j+1}/{len(max_iter_grid)}, max_iter: {max_iter}")
        lr = LogisticRegression(max_iter=max_iter, n_jobs=-1)
        lr.fit(X_train_pca, y_train)
        y_pred_val = lr.predict(X_val_pca)
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
print("Choosing hyperparameters for Naive Bayes...")
for i, n_components in enumerate(n_components_grid):
    print(f"Component: {i+1}/{len(n_components_grid)}, n_components: {n_components}")
    X_train_pca, X_val_pca = pca_cache[n_components]
    nb = GaussianNB()
    nb.fit(X_train_pca, y_train)
    y_pred_val = nb.predict(X_val_pca)
    acc = accuracy_score(y_val, y_pred_val)
    if acc > best_nb_acc:
        best_nb_acc = acc
        best_nb_params = {
            'n_components': n_components
        }

print(f"Best Naive Bayes params: {best_nb_params}, Validation Accuracy: {best_nb_acc:.4f}\n")

# Unir treino + validação para teste final
X_trainval = np.vstack([x_train_scaled, x_val_scaled])
y_trainval = np.concatenate([y_train, y_val])
# Ajustar max_iter para o tamanho do conjunto de treino + validação
best_log_params['max_iter'] *= int(len(X_trainval) / len(x_train_scaled))

# Teste final com os melhores parâmetros encontrados, aqui as imagens de teste são classificadas

# kNN euclidean
X_trainval_pca, X_test_pca = dimensionality_reduction(X_trainval, x_test_scaled, best_knn_euc_params['n_components'])
knn = KNeighborsClassifier(n_neighbors=best_knn_euc_params['n_neighbors'], metric='euclidean', n_jobs=-1)
knn.fit(X_trainval_pca, y_trainval)
y_pred_knn_euc = knn.predict(X_test_pca)
print("Test Accuracy kNN euclidean:", accuracy_score(y_test, y_pred_knn_euc))

# kNN manhattan
X_trainval_pca, X_test_pca = dimensionality_reduction(X_trainval, x_test_scaled, best_knn_man_params['n_components'])
knn = KNeighborsClassifier(n_neighbors=best_knn_man_params['n_neighbors'], metric='manhattan', n_jobs=-1)
knn.fit(X_trainval_pca, y_trainval)
y_pred_knn_man = knn.predict(X_test_pca)
print("Test Accuracy kNN manhattan:", accuracy_score(y_test, y_pred_knn_man))

# Logistic Regression
X_trainval_pca, X_test_pca = dimensionality_reduction(X_trainval, x_test_scaled, best_log_params['n_components'])
lr = LogisticRegression(max_iter=best_log_params['max_iter'], n_jobs=-1)
lr.fit(X_trainval_pca, y_trainval)
y_pred_log = lr.predict(X_test_pca)
print("Test Accuracy Logistic Regression:", accuracy_score(y_test, y_pred_log))

# Naive Bayes
X_trainval_pca, X_test_pca = dimensionality_reduction(X_trainval, x_test_scaled, best_nb_params['n_components'])
nb = GaussianNB()
nb.fit(X_trainval_pca, y_trainval)
y_pred_nb = nb.predict(X_test_pca)
print("Test Accuracy Naive Bayes:", accuracy_score(y_test, y_pred_nb))

# Resultados dos testes
results = []
results.append(f"Best kNN Euclidean params: {best_knn_euc_params}, Validation Accuracy: {best_knn_euc_acc:.4f}")
results.append(f"Best kNN Manhattan params: {best_knn_man_params}, Validation Accuracy: {best_knn_man_acc:.4f}")
results.append(f"Best Logistic Regression params: {best_log_params}, Validation Accuracy: {best_log_acc:.4f}")
results.append(f"Best Naive Bayes params: {best_nb_params}, Validation Accuracy: {best_nb_acc:.4f}\n")
results.append(f"Test Accuracy kNN Euclidean: {accuracy_score(y_test, y_pred_knn_euc):.4f}")
results.append(f"Test Accuracy kNN Manhattan: {accuracy_score(y_test, y_pred_knn_man):.4f}")
results.append(f"Test Accuracy Logistic Regression: {accuracy_score(y_test, y_pred_log):.4f}")
results.append(f"Test Accuracy Naive Bayes: {accuracy_score(y_test, y_pred_nb):.4f}")

# Nome do arquivo
output_dir = "tests"
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"TVT_{int(TRAIN_RATIO*100)}_{int(VALIDATION_RATIO*100)}_{int(TEST_RATIO*100)}_log.txt")

# Salvar resultados no arquivo
with open(filename, "w") as f:
    f.write("Tests Result :\n")
    f.write(f"Train Amount: {TRAIN_RATIO*100:.0f}% - Validation Amount: {VALIDATION_RATIO*100:.0f}% - Test Amount: {TEST_RATIO*100:.0f}%\n\n")
    for line in results:
        f.write(line + "\n")