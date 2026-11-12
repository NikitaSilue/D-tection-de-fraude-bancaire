# ============================================================
# PROJET SESSION IA - DÉTECTION DE FRAUDE BANCAIRE
# FAIT PAR NIKITA SILUE
# VERSION FINALE OPTIMISÉE
# ============================================================

# -------------------------
# IMPORTATION DES BIBLIOTHÈQUES
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA

# ============================================================
# 1. CHARGEMENT DU DATASET
# ============================================================
print("Chargement du dataset...")
df = pd.read_csv("C:\\Users\\Nikita\\Python\\ProjetIA\\archive\\train.csv")

print("Dimensions initiales :", df.shape)

# ============================================================
# 2. ÉCHANTILLONNAGE POUR RÉDUIRE LE TEMPS D'EXÉCUTION
# ============================================================
df = df.sample(n=50000, random_state=42)

print("Dimensions après échantillonnage :", df.shape)
print(df.head())
df.info()

# ============================================================
# 3. PRÉTRAITEMENT
# ============================================================

colonnes_a_supprimer = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "first",
    "last",
    "street",
    "trans_num"
]

df = df.drop(columns=colonnes_a_supprimer)

# Encodage robuste compatible Python 3.14
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) == False:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

print("\nTypes après encodage :")
print(df.dtypes)

x = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


# ============================================================
# VISUEL 1 — BIAIS OBSERVÉ (DÉSÉQUILIBRE DES CLASSES)
# ============================================================
plt.figure(figsize=(6,4))
sns.countplot(x='is_fraud', data=df)
plt.title("Distribution des classes : Fraude vs Non fraude")
plt.xlabel("Classe")
plt.ylabel("Nombre de transactions")
plt.show()


# ============================================================
# VISUEL 2 — VALEURS ABERRANTES (OUTLIERS)
# ============================================================
plt.figure(figsize=(8,4))
sns.boxplot(x=df["amt"])
plt.title("Valeurs aberrantes dans les montants de transaction")
plt.xlabel("Montant")
plt.show()

# ============================================================
# VISUEL 3 — FAISABILITÉ / SÉPARABILITÉ
# ============================================================
# Préparation PCA seulement pour visualisation
temp_x = df.drop("is_fraud", axis=1)
temp_scaled = StandardScaler().fit_transform(temp_x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(temp_scaled[:5000])
plt.figure(figsize=(8,5))
plt.scatter(x_pca[:,0], x_pca[:,1], c=df["is_fraud"][:5000], alpha=0.5)
plt.title("Visualisation de la séparabilité des classes")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.show()




scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Division Train / Test
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. ÉQUILIBRAGE AVEC SMOTE
# ============================================================
print("\nApplication de SMOTE...")
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

print("Distribution après SMOTE :")
print(pd.Series(y_train).value_counts())

# ============================================================
# FONCTION D'ÉVALUATION
# ============================================================
resultats = []

def evaluate_model(name, y_true, y_pred, temps):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n===== {name} =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("Temps d'exécution :", round(temps,2), "secondes")

    resultats.append([name, acc, prec, rec, f1, temps])

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion - " + name)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

# ============================================================
# 5. MODÈLES SCIKIT-LEARN
# ============================================================

# ----- Gradient Boosting -----
debut = time.time()
gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(x_train, y_train)
y_pred_gb = gb.predict(x_test)
fin = time.time()
evaluate_model("Gradient Boosting", y_test, y_pred_gb, fin-debut)

# ----- Logistic Regression -----
debut = time.time()
log = LogisticRegression(max_iter=1000)
log.fit(x_train, y_train)
y_pred_log = log.predict(x_test)
fin = time.time()
evaluate_model("Logistic Regression", y_test, y_pred_log, fin-debut)

# ----- Decision Tree -----
debut = time.time()
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
fin = time.time()
evaluate_model("Decision Tree", y_test, y_pred_dt, fin-debut)

# ----- Random Forest -----
debut = time.time()
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
fin = time.time()
evaluate_model("Random Forest", y_test, y_pred_rf, fin-debut)

# ============================================================
# 6. KNN FROM SCRATCH
# ============================================================

def knn_scratch(X_train, y_train, X_test, k=5):
    predictions = []

    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        k_indices = distances.argsort()[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions)

debut = time.time()
y_pred_knn_scratch = knn_scratch(x_train[:5000], y_train[:5000], x_test[:1000], k=5)
fin = time.time()
evaluate_model("KNN From Scratch", y_test[:1000], y_pred_knn_scratch, fin-debut)

# ============================================================
# 8. NAIVE BAYES FROM SCRATCH
# ============================================================

class NaiveBayesScratch:
    def __init__(self, smoothing=1e-9):
        self.smoothing = smoothing
        self.class_priors = None
        self.feature_stats = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.feature_stats = {}
        self.class_priors = {}
        
        n_samples, n_features = X.shape
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.feature_stats[cls] = {
                'mean': X_cls.mean(axis=0),
                'var': X_cls.var(axis=0) + self.smoothing
            }
            self.class_priors[cls] = len(X_cls) / n_samples

    def _calculate_likelihood(self, x, cls):
        mean = self.feature_stats[cls]['mean']
        var = self.feature_stats[cls]['var']
        # Gaussian probability density function
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.prod(numerator / denominator)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for cls in self.classes:
                prior = np.log(self.class_priors[cls])
                likelihood = np.log(self._calculate_likelihood(x, cls) + self.smoothing)
                posteriors[cls] = prior + likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

debut = time.time()
nb_scratch = NaiveBayesScratch(smoothing=1e-9)
nb_scratch.fit(x_train[:10000], np.array(y_train[:10000]))
y_pred_nb_scratch = nb_scratch.predict(x_test[:2000])
fin = time.time()
evaluate_model("Naive Bayes From Scratch", y_test[:2000], y_pred_nb_scratch, fin-debut)

# ============================================================
# 9. TABLEAU FINAL COMPARATIF
# ============================================================

df_resultats = pd.DataFrame(resultats, columns=[
    "Modèle", "Accuracy", "Precision", "Recall", "F1 Score", "Temps"
])

print("\n================ TABLEAU FINAL ==================")
print(df_resultats)

plt.figure(figsize=(10,5))
sns.barplot(x="Modèle", y="F1 Score", data=df_resultats)
plt.xticks(rotation=45)
plt.title("Comparaison des F1 Scores")
plt.show()