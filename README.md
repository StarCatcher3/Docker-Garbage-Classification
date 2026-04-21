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
- BERGEN Kirtika
- REPLOGLE John
- TE Mathis
- ROBILLIARD Diane

---

## Docker

La stack Docker suit une architecture à **5 services**, **3 volumes** et **3 réseaux** :

### Services
- `spark-service` : transformation des images brutes en Parquet (one-shot)
- `keras-service` : entraînement du modèle CNN (one-shot)
- `db-service` : stockage et accès aux données Parquet via API REST
- `api-service` : réception des images et retour des prédictions
- `streamlit-front` : interface utilisateur

### Volumes
- `./data` → `/app/data` — images brutes, input, archive, prédictions
- `./parquet` → `/app/data` (db-service) — fichiers Parquet train/test
- `./models` → `/app/models` — modèle `final_CNN.keras` et logs TensorBoard

### Réseaux
- `front-network` : `streamlit-front` ↔ `api-service`
- `transform-network` : `spark-service` → `db-service` (POST /save)
- `model-network` : `db-service` → `keras-service` (GET /data)

---

### 1. Lancer les services persistants

```bash
docker compose up --build -d api-service streamlit-front db-service
```

Le front est disponible sur `http://localhost:8501`.

---

### 2. Transformer les images en Parquet

```bash
docker compose run --rm spark-service data/data_transformation.py
```

PySpark lit `./data/train` et `./data/test`, transforme les images en pixels 64×64 grayscale + encodage one-hot, et envoie les données à `db-service` via `transform-network`.

---

### 3. Entraîner le modèle

```bash
docker compose run --rm keras-service models/training.py
```

Keras récupère les données depuis `db-service` via `model-network`, entraîne le CNN, et sauvegarde `final_CNN.keras` dans `./models`.

---

### 4. Lancer une prédiction

Déposer une image dans `./data/input/` puis appeler l'API :

```bash
curl -X POST http://localhost:8000/upload -F "file=@photo.jpg"
```

Ou directement depuis l'interface Streamlit sur `http://localhost:8501`.

---

### 5. Variables d'environnement (`.env`)

| Variable | Description |
|---|---|
| `GARBAGE_RAW_DATA_DIR` | Répertoire des images brutes (`/app/data`) |
| `GARBAGE_MODEL_DIR` | Répertoire du modèle (`/app/models`) |
| `GARBAGE_MODEL_PATH` | Chemin complet du modèle `.keras` |
| `SPARK_MASTER` | Valeur Spark, par défaut `local[*]` |

---

### 6. Organisation des flux

- `spark-service` lit les images depuis le volume `./data`
- `spark-service` envoie les Parquet à `db-service` via `transform-network` (POST /save)
- `db-service` stocke les Parquet dans le volume `./parquet`
- `keras-service` demande les données à `db-service` via `model-network` (GET /data)
- `keras-service` sauvegarde le modèle dans le volume `./models`
- `api-service` reçoit les images via `front-network` (POST /upload)
- `api-service` retourne les prédictions à `streamlit-front` (POST /get, GET /data)