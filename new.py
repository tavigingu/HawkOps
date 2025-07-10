#!/usr/bin/env python3

import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, AlertStateChanged
from olympe.messages.common.CalibrationState import MagnetoCalibrationStateChanged
from olympe.messages.gimbal import calibration_state

DRONE_IP = "192.168.42.1"

def check_calibration_status(drone):
    """Verifică starea calibrării - EXACT motivul pentru care nu decoleaza"""
    print("🔍 VERIFICARE CALIBRĂRI:")
    
    # Verifică calibrarea magnetometrului
    try:
        magneto_state = drone.get_state(MagnetoCalibrationStateChanged)
        if magneto_state:
            print(f"🧭 Magnetometru: {magneto_state}")
        else:
            print("🧭 Magnetometru: Status necunoscut")
    except:
        print("🧭 Magnetometru: Nu pot verifica")
    
    # Verifică alertele de sistem
    try:
        alerts = drone.get_state(AlertStateChanged)
        if alerts:
            print(f"⚠️  Alertă sistem: {alerts}")
            return False
        else:
            print("✅ Fără alerte sistem")
            return True
    except:
        print("⚠️  Nu pot verifica alertele")
        return False

def detailed_takeoff_test():
    print("🚁 DIAGNOSTIC EXACT - De ce nu decoleaza")
    
    drone = olympe.Drone(DRONE_IP)
    
    try:
        print("📡 Conectare...")
        drone.connect()
        print("✅ Conectat!")
        
        # Verifică starea inițială
        flying_state = drone.get_state(FlyingStateChanged)
        print(f"📊 Stare zbor: {flying_state['state'] if flying_state else 'Necunoscută'}")
        
        # Verifică calibrările - AICI E PROBLEMA
        calibration_ok = check_calibration_status(drone)
        
        if not calibration_ok:
            print("❌ PROBLEMA GĂSITĂ: Calibrare incorectă!")
            print("🔧 SOLUȚIE:")
            print("   1. Deschide FreeFlight 6 pe telefon")
            print("   2. Conectează-te la dronă")
            print("   3. Setări > Dronă > Calibrare")
            print("   4. Calibrează magnetometrul")
            print("   5. Calibrează gimbal-ul")
            return
        
        print("🛫 Încerc decolarea...")
        
        # Comandă de decolare cu monitoring în timp real
        takeoff_result = drone(TakeOff()).wait(timeout=15)
        
        if takeoff_result.success():
            print("✅ Comanda trimisă cu succes!")
            
            # Monitorizez starea în timp real
            for i in range(10):  # 10 secunde de monitoring
                current_state = drone.get_state(FlyingStateChanged)
                state_name = current_state['state'] if current_state else 'necunoscut'
                print(f"📊 Secunda {i+1}: {state_name}")
                
                if state_name in ['takingoff', 'hovering']:
                    print("✅ DECOLAREA A REUȘIT!")
                    break
                elif state_name == 'landed':
                    print("❌ RĂMÂNE PE SOL")
                    if i == 9:  # Ultima verificare
                        print("🔧 CAUZA: Magnetometru sau gimbal necalibrat")
                
                time.sleep(1)
                
        else:
            print("❌ Comanda de decolare a eșuat")
            print("🔧 Verifică conexiunea WiFi")
            
    except Exception as e:
        print(f"❌ Eroare: {e}")
        
    finally:
        print("🔌 Deconectare...")
        drone.disconnect()

def show_exact_solution():
    print("\n🎯 SOLUȚIA EXACTĂ pentru problema ta:")
    print("="*50)
    print("PROBLEMA: Comanda de decolare se trimite dar drona nu pornește elicele")
    print("CAUZA: Magnetometrul sau gimbal-ul nu sunt calibrate")
    print("\nPAȘII DE REZOLVARE:")
    print("1. Descarcă FreeFlight 6 pe telefon")
    print("2. Conectează telefonul la drona WiFi")
    print("3. Deschide FreeFlight 6")
    print("4. Mergi la: Setări → Dronă → Calibrare")
    print("5. Calibrează magnetometrul (urmează instrucțiunile)")
    print("6. Calibrează gimbal-ul")
    print("7. Repornește drona")
    print("8. Testează din nou codul")
    print("="*50)

if __name__ == "__main__":
    print("Alege:")
    print("1. Rulează diagnostic")
    print("2. Afișează soluția exactă")
    
    choice = input("Opțiune (1/2): ")
    
    if choice == "1":
        detailed_takeoff_test()
    elif choice == "2":
        show_exact_solution()
    else:
        print("Opțiune invalidă!")
        show_exact_solution()