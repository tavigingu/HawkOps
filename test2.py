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
        
        # Verifică starea inițială
        print("🔍 Verificare stare inițială...")
        initial_state = drone.get_state(FlyingStateChanged)
        print(f"📊 Stare inițială: {initial_state['state'] if initial_state else 'Necunoscută'}")
        
        # Așteaptă 3 secunde pentru stabilizare
        print("⏳ Aștept stabilizarea dronei...")
        time.sleep(3)
        
        print("🛫 Decolare...")
        # Comandă de decolare 
        takeoff_result = drone(TakeOff()).wait(timeout=10)
        
        if takeoff_result.success():
            print("✅ Comandă decolare trimisă cu succes!")
            
            # Verifică starea după decolare
            time.sleep(2)
            current_state = drone.get_state(FlyingStateChanged)
            print(f"📊 Stare după decolare: {current_state['state'] if current_state else 'Necunoscută'}")
            
            if current_state and current_state['state'] in ['takingoff', 'hovering']:
                print("✅ Decolat!")
                
                # Așteaptă 5 secunde în zbor
                print("⏱️ Zbor stationar 5 secunde...")
                time.sleep(5)
                
                print("🛬 Aterizare...")
                landing_result = drone(Landing()).wait(timeout=10)
                
                if landing_result.success():
                    print("✅ Aterizat!")
                else:
                    print("❌ Aterizarea a eșuat")
            else:
                print("❌ Drona nu a decolat - rămâne în starea 'landed'")
                print("🔧 Cauze posibile:")
                print("   - Baterie prea descărcată")
                print("   - Suprafața nu este detectată ca sigură")
                print("   - Calibrarea IMU/magnetometru")
                print("   - Spațiu insuficient detectat de senzori")
        else:
            print("❌ Comanda de decolare a eșuat")
            
        print("🎉 Test de zbor finalizat!")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        print("⚠️ Verificări de bază:")
        print("   - Drona pe suprafață plană")
        print("   - Baterie >20%")
        print("   - Spațiu liber minim 2m")
        
    finally:
        print("🔌 Deconectare...")
        drone.disconnect()
        print("👋 Gata!")

if __name__ == "__main__":
    simple_flight_test()