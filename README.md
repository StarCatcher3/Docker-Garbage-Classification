# 🗑️ Garbage Detection Classifier

Ce projet propose une solution complète de **classification de déchets** afin d’améliorer le tri dans les différentes catégories de poubelles.

Il repose sur une architecture distribuée combinant :

- **PySpark** pour l’ingénierie des données
- **Keras (TensorFlow)** pour l’entraînement du modèle CNN
- **Streamlit** pour la visualisation et l’affichage des prédictions

---

## 📊 Dataset

Les images proviennent du dataset Kaggle :

**Garbage Detection – 6 Waste Categories**  
Auteur : Viswa Prakash  
Lien : https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection

Le dataset contient des images classées en différentes catégories de déchets (plastique, carton, verre, métal, etc.).

---

## 🎯 Objectifs

- Mettre en place un pipeline complet Data Engineering → Machine Learning → Visualisation
- Exploiter PySpark pour le traitement distribué d’images
- Implémenter un CNN pour la classification multi-classe
- Déployer un dashboard interactif

---

## ✏️ Groupe de travail

- BAHOURI Mohamed Elyes
- REPLOGLE John
- TE Mathis
- ROBILLIARD Diane

---

## Docker

La stack Docker suit une architecture à trois services et trois volumes logiques :

- `spark-service` : transformation des images et prédiction
- `keras-service` : entraînement du modèle
- `streamlit-front` : interface utilisateur Streamlit
- volume 1 : images brutes
- volume 2 : fichiers Parquet
- volume 3 : modèle `.keras`

### 1. Volumes utilisés

- `./data` vers `/app/data`
- `./parquet` vers `/app/database`
- `./models` vers `/app/models`

### 2. Lancer l'architecture complète

```bash
docker compose up --build -d
```

Le front est disponible sur `http://localhost:8501`.

### 3. Lancer les jobs Spark

Transformation des images brutes en Parquet :

```bash
docker compose run spark-service
```

Entraînement du modèle et sauvegarde dans le volume modèles :

```bash
docker compose run keras-service
```

Prédiction sur les images déposées dans `data/input` :

```bash
docker compose run spark-service data/run_prediction.py
```

### 4. Variables d'environnement utiles

- `GARBAGE_RAW_DATA_DIR` : répertoire des images brutes
- `GARBAGE_PARQUET_DIR` : répertoire des sorties Parquet
- `GARBAGE_MODEL_PATH` : chemin du modèle `.keras`
- `SPARK_MASTER` : valeur Spark, par défaut `local[*]`

### 5. Organisation cible

- `spark-service` lit les images dans le volume 1
- `spark-service` écrit les jeux de données Parquet dans le volume 2
- `keras-service` lit les jeux de données Parquet dans le volume 2
- `keras-service` sauvegarde le modèle dans le volume 3
- `spark-service` lit le modèle dans le volume 3
- `streamlit-front` dépose les images à prédire dans le volume 1
- `spark-service` transforme et prédit les images dans le volume 1
- `spark-service` écrit les prédictions Parquet dans le volume 1
- `streamlit-front` lit les prédictions Parquet dans le volume 1
