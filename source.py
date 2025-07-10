#!/usr/bin/env python3

import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

# Adresa IP a dronei (default pentru Anafi)
DRONE_IP = "192.168.42.1"

def test_connection():
    print("🚁 Încercăm conectarea la Parrot Anafi...")
    
    # Creează instanța dronei
    drone = olympe.Drone(DRONE_IP)
    
    try:
        # Conectează-te la dronă
        print("📡 Conectare în curs...")
        drone.connect()
        print("✅ Conectat cu succes!")
        
        # Verifică starea dronei
        print("🔍 Verificare stare dronă...")
        state = drone.get_state(FlyingStateChanged)
        print(f"📊 Starea dronei: {state}")
        
        # Așteaptă 2 secunde
        time.sleep(2)
        
        print("✅ Test de conectare reușit!")
        
    except Exception as e:
        print(f"❌ Eroare la conectare: {e}")
        print("💡 Verifică că:")
        print("   - Drona este pornită")
        print("   - Ești conectat la WiFi-ul dronei")
        print("   - IP-ul este corect (192.168.42.1)")
        
    finally:
        # Deconectează-te
        print("🔌 Deconectare...")
        drone.disconnect()
        print("👋 Deconectat!")

if __name__ == "__main__":
    test_connection()