import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Reduccion de dimensionalidad : PCA
from sklearn.decomposition import PCA 

from sklearn.model_selection import train_test_split
# Agreguemos lo necesario para crear un randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz
# GridSearch 
from sklearn.model_selection import GridSearchCV



import os 
os.chdir("D:/ML_Spyder/Class6")

df = pd.read_csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/CreditCard_Customer.csv")

# Proprocesamiento 

# Usamos fillna para rellenar los valores faltantes
df.MINIMUM_PAYMENTS = df.MINIMUM_PAYMENTS.fillna(df.MINIMUM_PAYMENTS.mean())
df.CREDIT_LIMIT = df.CREDIT_LIMIT.fillna(df.CREDIT_LIMIT.mean())


# Creemos un nuevo dataframe sin considerar la columna CUST_ID
df_Num = df.drop(["CUST_ID"], axis = "columns")
df_Num.columns
df_Num.info()

# Escalamos los datos a (0,1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_Num)

# Construyamos un dataframe 
df_sc = pd.DataFrame(X_scaled)
# 
# Asignemos a este nuevo dataframa los nombres de las columnas 
df_sc.columns = df_Num.columns


# PCA : Reduccion de dimensionalidad 
# Para poder ver nuestros cluster 

ModelPCA = PCA(n_components=2 ) 
ModelPCA.fit(df_sc)
df_PCA = ModelPCA.transform(df_sc)



# %% KMeans con k = 3 
ModelKMeans = KMeans(n_clusters = 3 ,
                     max_iter = 1000,
                     n_init =20)
# 
# Este ModelKMeans lo deseo ajustar con el dataframe proveniente del 
# ModelPCA
ModelKMeans.fit(df_PCA)


# Visualicemos como quedo nuestro Kmeans 
plt.scatter(df_PCA[:, 0], df_PCA[:, 1],
            c = ModelKMeans.labels_)
plt.xlabel("PCA_1")
plt.ylabel("PCA_2")
plt.show()


# Usemos estas etiquetas para etiquetar nuestro dataset
df_sc["cluster"] = ModelKMeans.labels_

# Usemos las etiquetas provenientes del KMeans para nuestro 
# conjunto de datos original 
df_Num["cluster"] = ModelKMeans.labels_

# %% Estadistica descriptiva de los datos ahora etiquetados 
# 

df_Num.info()
# 
# Transformamos la variable cluster a una variable de tipo str
# para considerarla como una variable cualitativa
df_Num.cluster = df_Num.cluster.astype("str")

# Diagrama de barras para la variable cluster 
sns.catplot(x = "cluster",
            # hue = "Otra_variable_cualitativa",
            kind = "count",
            data = df_Num)
plt.savefig("DiagramaBarras_cluster.png")

# Histograma de BALANCE vs cluster
sns.displot(x = "BALANCE",
            hue = "cluster",
            data = df_Num,
            kde = True)

sns.histplot(x = "BALANCE",
            hue = "cluster",
            bins = 20,
            data = df_Num,
            kde = False)


# Creamos una carpeta para todos los histplot
os.makedirs("histplots")
# Repliquemos estos histplot para todas las columnas 
cols = df_Num.columns[:-1]
for c in cols:
    plt.figure(figsize = (10,10))
    sns.histplot(x = c,
                hue = "cluster",
                bins = 20,
                data = df_Num,
                kde = False)
    Figname = "histplots/variable_" + c + ".png"
    plt.savefig(Figname,dpi = 300)

# Boxplots
sns.boxplot(x = "cluster",
            y = "BALANCE",
            data = df_Num)

# Creamos una carpeta para todos los boxplots
os.makedirs("boxplots")
# 
for c in cols:
    plt.figure(figsize = (10,10))
    sns.boxplot(x = "cluster",
                y = c,
                # hue = "otra_variable_cualitativa",
                data = df_Num)
    Figname = "boxplots/variable_" + c + ".png"
    plt.savefig(Figname, dpi = 300)

# Diagramas de dispersion : scatterplot
# COmparemos CREDIT_LIMIT vs BALANCE en funcion de cluster
plt.figure(figsize = (10,10))
sns.scatterplot(x = "CREDIT_LIMIT",
                y = "BALANCE",  
                hue = "cluster",
                data = df_Num)
plt.savefig("DiagramaDispersion_CreditLimit_balance_cluster.png",
            dpi = 300)
# 

plt.figure(figsize=(12,12))
sns.scatterplot(df_Num,
                x = "CREDIT_LIMIT",
                y = "PAYMENTS",
                hue = "cluster")
plt.savefig("Diagramapormi.png",
            dpi = 400)



# plt.figure(figsize = (10,10))
# sns.scatterplot(x = "CREDIT_LIMIT",
#                 y = "PAYMENTS",
#                 hue = "cluster",
#                 data = df_Num)
# plt.savefig("DiagramaDispersion_CreditLimit_payments_cluster.png",
#             dpi = 300)

# %% Guardemos el dataset etiquetado 

df_Num.to_csv("ConsumerLabeld.csv")

# %% Busquemos predecir la variable dependiente : cluster 

data = pd.read_csv("ConsumerLabeld.csv")

# Separar las variables independientes de la variable dependiente
X = data.drop("cluster", axis= 1)
y = data.cluster

# Particionamos los datos 
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   test_size = 0.15)

# Creamos un modelo de tipo random forest de clasificacion
Model1 = RandomForestClassifier()
Model1.fit(X_train, y_train)
Model1.score(X_test, y_test)
# aprox 98%
#0.9843633655994043
# 
Model1.get_params()
help(RandomForestClassifier)


# Construyamos los pronosticos del X_test
yForecast = Model1.predict(X_test)

# Calculemos el accuracy_score
acc_score = metrics.accuracy_score(y_test, yForecast)
# aprox 98%
# 0.9843633655994043


# GridSearch 
DictHiperParam = {"n_estimators" : [80, 90, 110, 120, 200, 300, 500],
                  "criterion" : ["gini", "entropy"],
                  "max_depth" : [5,10,15, 25, 50, 75, 100],
                  "min_samples_split" : [0.1,0.25,0.35,0.55,0.7,0.75,0.8],
                  "min_samples_leaf" : np.linspace(0.05, 0.75, 25),
                  "ccp_alpha" : [1,5,15,30,70,99],
                  "bootstrap" : [False, True]}

RandomForest_GS = GridSearchCV(estimator = RandomForestClassifier(), 
                               param_grid = DictHiperParam,
                               cv = 5,
                               verbose = 4,
                               n_jobs = 3,
                               scoring = "accuracy")

# Ajustemos el gridsearch 
MejorBosque = RandomForest_GS.fit(X_train, y_train)





