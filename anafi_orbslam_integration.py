#!/usr/bin/env python3
"""
Integrare Parrot Anafi cu ORB-SLAM3
Captează stream video de la dronă și îl procesează cu ORB-SLAM3
"""

import olympe
import cv2
import numpy as np
import subprocess
import os
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged

class AnafiORBSlam:
    def __init__(self):
        self.drone_ip = "192.168.42.1"  # IP-ul implicit al Anafi
        self.drone = None
        self.is_streaming = False
        self.output_dir = "/tmp/orbslam_frames"
        
        # Creează directorul pentru frame-uri
        os.makedirs(self.output_dir, exist_ok=True)
        
    def connect_drone(self):
        """Conectează la drona Parrot Anafi"""
        try:
            self.drone = olympe.Drone(self.drone_ip)
            self.drone.connect()
            print(f"✅ Conectat la drona Anafi pe {self.drone_ip}")
            return True
        except Exception as e:
            print(f"❌ Eroare la conectarea la dronă: {e}")
            return False
    
    def start_video_stream(self):
        """Pornește stream-ul video de la dronă"""
        try:
            # Configurează stream-ul video
            self.drone.streaming.set_callbacks(
                raw_cb=self.video_frame_callback
            )
            self.drone.streaming.start()
            self.is_streaming = True
            print("📹 Stream video pornit")
            return True
        except Exception as e:
            print(f"❌ Eroare la pornirea stream-ului: {e}")
            return False
    
    def video_frame_callback(self, yuv_frame):
        """Callback pentru procesarea frame-urilor video"""
        if not self.is_streaming:
            return
            
        try:
            # Convertește YUV la BGR
            cv2_cvt_color_flag = {
                olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
            }
            
            # Obține frame-ul în format OpenCV
            cv2_frame = yuv_frame.as_cv2_frame(
                cv2_cvt_color_flag=cv2_cvt_color_flag[yuv_frame.format()]
            )
            
            # Salvează frame-ul cu timestamp
            timestamp = int(time.time() * 1000000)  # microsecunde
            filename = f"{self.output_dir}/{timestamp:016d}.png"
            cv2.imwrite(filename, cv2_frame)
            
            # Afișează frame-ul (opțional)
            cv2.imshow("Anafi Video Stream", cv2_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_streaming()
                
        except Exception as e:
            print(f"❌ Eroare la procesarea frame-ului: {e}")
    
    def stop_streaming(self):
        """Oprește stream-ul video"""
        if self.is_streaming:
            self.drone.streaming.stop()
            self.is_streaming = False
            cv2.destroyAllWindows()
            print("⏹️ Stream video oprit")
    
    def create_timestamps_file(self):
        """Creează fișierul cu timestamp-uri pentru ORB-SLAM3"""
        image_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])
        
        timestamps_file = f"{self.output_dir}/timestamps.txt"
        with open(timestamps_file, 'w') as f:
            for img_file in image_files:
                # Extrage timestamp-ul din numele fișierului
                timestamp = int(img_file.split('.')[0])
                # Convertește la secunde cu 6 zecimale
                timestamp_sec = timestamp / 1000000.0
                f.write(f"{timestamp_sec:.6f}\n")
        
        print(f"📝 Fișier timestamps creat: {timestamps_file}")
        return timestamps_file
    
    def run_orbslam3(self, config_file="TUM1.yaml"):
        """Rulează ORB-SLAM3 cu imaginile capturate"""
        timestamps_file = self.create_timestamps_file()
        
        # Comanda pentru ORB-SLAM3
        orbslam_cmd = [
            "sudo", "docker", "exec", "-it", "orbslam_container",
            "/ORB_SLAM3/Examples/Monocular/mono_tum",
            "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            f"/ORB_SLAM3/Examples/Monocular/{config_file}",
            self.output_dir
        ]
        
        print("🚀 Pornesc ORB-SLAM3...")
        try:
            subprocess.run(orbslam_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Eroare la rularea ORB-SLAM3: {e}")
    
    def takeoff(self):
        """Decolează drona"""
        if self.drone and self.drone.is_connected():
            self.drone(TakeOff()).wait()
            print("🚁 Drona a decolat")
    
    def land(self):
        """Aterizează drona"""
        if self.drone and self.drone.is_connected():
            self.drone(Landing()).wait()
            print("🛬 Drona a aterizat")
    
    def disconnect(self):
        """Deconectează de la dronă"""
        if self.drone:
            self.stop_streaming()
            self.drone.disconnect()
            print("🔌 Deconectat de la dronă")


def main():
    """Funcția principală"""
    anafi_slam = AnafiORBSlam()
    
    try:
        # Conectează la dronă
        if not anafi_slam.connect_drone():
            return
        
        # Pornește stream-ul video
        if not anafi_slam.start_video_stream():
            return
        
        print("\n📋 Comenzi disponibile:")
        print("  't' - Decolează")
        print("  'l' - Aterizează") 
        print("  's' - Oprește stream și rulează ORB-SLAM3")
        print("  'q' - Ieși")
        
        # Loop principal
        while True:
            cmd = input("\nIntroduceți comanda: ").lower().strip()
            
            if cmd == 't':
                anafi_slam.takeoff()
            elif cmd == 'l':
                anafi_slam.land()
            elif cmd == 's':
                anafi_slam.stop_streaming()
                anafi_slam.run_orbslam3()
                break
            elif cmd == 'q':
                break
            else:
                print("❓ Comandă necunoscută")
    
    except KeyboardInterrupt:
        print("\n⏹️ Oprire prin Ctrl+C")
    
    finally:
        anafi_slam.disconnect()


if __name__ == "__main__":
    main()