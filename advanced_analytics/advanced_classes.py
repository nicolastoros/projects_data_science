import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#Clases utiles

#Modelos No Supervisados
class ClusteringModels_v1:
    def __init__(self, X, model_type=None, n_clusters=None):
        self.X = X
        self.model_type = model_type
        self.n_clusters = n_clusters
    def fit_predict(self):
        if self.model_type == "kmeans":
            model = KMeans(n_clusters=self.n_clusters)
        elif self.model_type == "dbscan":
            model = DBSCAN()
        elif self.model_type == "aggclust":
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError("Invalid model type")
        return model.fit_predict(self.X)
class OptimalK:
    def __init__(self, max_k=10):
        self.max_k = max_k
        self.sum_of_squared_distances = []

    def fit(self, data):
        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans = kmeans.fit(data)
            self.sum_of_squared_distances.append(kmeans.inertia_)

    def plot_elbow(self):
        plt.plot(range(1, self.max_k + 1), self.sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def optimal_k(self):
        return np.argmin(np.diff(self.sum_of_squared_distances)) + 1
class ClusteringModels:
    def __init__(self, X, models, metrics, n_clusters=None):
        self.X = X
        self.models = models
        self.metrics = metrics
        self.n_clusters = n_clusters

    def fit_predict(self, model_type, n_clusters=None):
        if model_type == 'KMeans':
            model = KMeans(n_clusters=n_clusters)
        elif model_type == 'AgglomerativeClustering':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif model_type == 'DBSCAN':
            model = DBSCAN()
        else:
            raise ValueError("Invalid model type")
        return model.fit_predict(self.X)

    def evaluate_models(self):
        results = []
        for model_type in self.models:
            if self.n_clusters:
                labels = self.fit_predict(model_type, self.n_clusters)
            else:
                labels = self.fit_predict(model_type)
            silhouette = silhouette_score(self.X, labels) if 'silhouette' in self.metrics else None
            calinski = calinski_harabasz_score(self.X, labels) if 'calinski_harabasz' in self.metrics else None
            davies = davies_bouldin_score(self.X, labels) if 'davies_bouldin' in self.metrics else None
            results.append({'model': model_type, 'silhouette': silhouette, 'calinski_harabasz': calinski,
                            'davies_bouldin': davies})
        return pd.DataFrame(results)

    def select_best_model(self, metric='silhouette'):
        results = self.evaluate_models()
        best_model = results.loc[results[metric].idxmax(), 'model']
        return best_model
class DimensionalityReductionModels:
    def __init__(self, X, y=None, n_components=2):
        self.X = X
        self.y = y
        self.n_components = n_components
        self.models = {
            'PCA': PCA(n_components=self.n_components),
            'ICA': FastICA(n_components=self.n_components),
            'NMF': NMF(n_components=self.n_components),
            't-SNE': TSNE(n_components=self.n_components)
        }

    def fit(self, model_name):
        self.model = self.models[model_name]
        self.reduced_X = self.model.fit_transform(self.X)
        return self.reduced_X

#Modelos Supervisados
class OptimalTrainTestSplit:
    def __init__(self, model, X, y, n_splits=5, random_state=0):
        self.model = model
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state

    def find_optimal_split(self):
        kf = KFold(n_splits=self.n_splits, random_state=self.random_state)
        best_split = (None, np.inf)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            if mse < best_split[1]:
                best_split = (train_index, test_index, mse)
        return best_split
class RegressionModels:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = [
            LinearRegression(),
            Lasso(),
            Ridge(),
            ElasticNet(),
            RandomForestRegressor(),
            GradientBoostingRegressor()
        ]

    def train_test_split(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        results = []
        for model in self.models:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results.append((model, score))
        return results

    def select_best_model(self, results):
        best_score = -np.inf
        best_model = None
        for model, score in results:
            if score > best_score:
                best_score = score
                best_model = model
        return best_model

    def stacked_regression(self, X_train, X_test, y_train, y_test):
        estimators = [
            ('lr', LinearRegression()),
            ('lasso', Lasso()),
            ('ridge', Ridge()),
            ('en', ElasticNet()),
            ('rf', RandomForestRegressor()),
            ('gb', GradientBoostingRegressor())
        ]
        stacked_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression()
        )
        stacked_regressor.fit(X_train, y_train)
        score = stacked_regressor.score(X_test, y_test)
        return stacked_regressor, score


import numpy as np

# Generamos los datos de entrada
X, y = make_blobs(n_samples=300, centers=4, random_state=0)
data = make_blobs(n_samples=200, centers=4, random_state=0)[0]

ok = OptimalK()
ok.fit(data)
ok.plot_elbow()
k = ok.optimal_k()
print("El número óptimo de clusters es:", k)



clustering_models = ClusteringModels(n_clusters=4, models=['KMeans', 'DBSCAN', 'AgglomerativeClustering'],
                                     metrics=['silhouette','davies_bouldin','calinski_harabasz'], X=X)
#Evaluacion de los modelos
clustering_models.evaluate_models()
# Escogemos el mejor modelo
best_model = clustering_models.select_best_model()
# Imprimimos el modelo seleccionado
print("El mejor modelo es: ", best_model)


# Crear una instancia de la clase
clustering = ClusteringModels_v2(X)

# Ajustar los modelos de clustering
clustering.fit_models()

# Evaluar los modelos de clustering
clustering.evaluate_models()

# Obtener el mejor modelo de clustering
best_model

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

# Cargar los datos
digits = load_digits()
X = digits.data
y = digits.target

# Instanciar la clase
dr_model = DimensionalityReductionModels(X)

# Escoger el modelo y hacer la reducción de dimensionalidad
reduced_X = dr_model.fit('t-SNE')






from sklearn.linear_model import LinearRegression
import numpy as np

# Generamos datos de prueba
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

# Instanciamos el modelo y la clase
model = LinearRegression()
split = OptimalTrainTestSplit(model, X, y)

# Encontramos el split óptimo
train_index, test_index, mse = split.find_optimal_split()









