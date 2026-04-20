#!/bin/bash

echo "=== Lancement de la prédiction automatique d'images toutes les 5 minutes ==="
while true; do
    python src/run_prediction.py
    sleep 1
done
