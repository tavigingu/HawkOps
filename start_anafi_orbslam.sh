#!/bin/bash

echo "🚁 Pornire sistem Anafi + ORB-SLAM3"

# Activează mediul virtual
source ~/anafi_env/bin/activate

# Verifică conexiunea la dronă
echo "🔍 Verificare conexiune la Anafi (192.168.42.1)..."
ping -c 3 192.168.42.1

if [ $? -eq 0 ]; then
    echo "✅ Drona este accesibilă"
    echo "🎬 Pornesc aplicația..."
    python3 anafi_orbslam_integration.py
else
    echo "❌ Nu pot accesa drona. Verifică conexiunea WiFi."
    echo "💡 Conectează-te la rețeaua WiFi a dronei (Anafi-XXXXX)"
fi