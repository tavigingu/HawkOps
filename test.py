#!/usr/bin/env python3

import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

DRONE_IP = "192.168.42.1"

def simple_flight_test():
    print("🚁 Test simplu de zbor - Parrot Anafi")
    
    drone = olympe.Drone(DRONE_IP)
    
    try:
        print("📡 Conectare...")
        drone.connect()
        print("✅ Conectat!")
        
        print("🛫 Decolare...")
        # Comandă de decolare și așteaptă să se execute
        drone(TakeOff()).wait().success()
        print("✅ Decolat!")
        
        # Așteaptă 5 secunde în zbor
        print("⏱️ Zbor stationar 5 secunde...")
        time.sleep(5)
        
        print("🛬 Aterizare...")
        # Comandă de aterizare și așteaptă să se execute
        drone(Landing()).wait().success()
        print("✅ Aterizat!")
        
        print("🎉 Test de zbor reușit!")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        print("⚠️ Asigură-te că:")
        print("   - Drona este pe o suprafață plană")
        print("   - Ai suficient spațiu pentru decolare")
        print("   - Bateria este încărcată")
        
    finally:
        print("🔌 Deconectare...")
        drone.disconnect()
        print("👋 Gata!")

if __name__ == "__main__":
    simple_flight_test()