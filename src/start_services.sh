#!/bin/bash

echo "=== Lancement de la prédiction automatique d'images (verifie data/input toutes les 15s) ==="
while true; do
    python src/run_prediction.py
    sleep 15
done &

echo "=== Lancement de Streamlit ==="
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
