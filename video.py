#!/usr/bin/env python3

import olympe
import time
import threading
import cv2
import numpy as np
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.camera import (
    set_camera_mode,
    start_recording,
    stop_recording,
    set_streaming_mode,
)

DRONE_IP = "192.168.42.1"

class ColorVideoStream:
    def __init__(self, drone):
        self.drone = drone
        self.video_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        
    def start_video_stream(self):
        """Pornește streaming-ul video cu conversii de culori îmbunătățite"""
        try:
            print("📹 Configurare streaming video cu suport culori...")
            
            # Configurează camera pentru cea mai bună calitate
            self.drone(set_camera_mode(cam_id=0, value="recording")).wait()
            
            # Încearcă să seteze modul de streaming
            try:
                self.drone(set_streaming_mode(cam_id=0, value="low_latency")).wait()
                print("📹 Mod streaming low_latency setat")
            except:
                print("⚠️ Nu s-a putut seta modul streaming, continuă cu setările default")
            
            # Configurează callback-urile cu focus pe frame-uri de calitate
            self.drone.streaming.set_callbacks(
                raw_cb=self._color_frame_callback,
                h264_cb=self._h264_frame_callback,
                start_cb=self._start_callback,
                end_cb=self._end_callback,
                flush_raw_cb=self._flush_callback,
            )
            
            # Pornește streaming-ul
            self.drone.streaming.start()
            self.video_running = True
            print("📹 Streaming video pornit cu suport culori!")
            
            return True
            
        except Exception as e:
            print(f"❌ Eroare la pornirea video: {e}")
            return False
    
    def _color_frame_callback(self, frame):
        """Callback pentru frame-uri cu conversii de culori îmbunătățite"""
        try:
            if frame is not None and self.video_running:
                self.frame_count += 1
                
                # Informații despre frame
                info = frame.info()
                print(f"📺 Frame {self.frame_count}: {info}")
                
                # Extrage datele frame-ului
                frame_data = frame.as_ndarray()
                if frame_data is None:
                    print("⚠️ Nu s-au putut extrage datele frame-ului")
                    return
                
                print(f"📺 Frame data - Shape: {frame_data.shape}, dtype: {frame_data.dtype}")
                
                # Încearcă multiple metode de conversie pentru culori
                rgb_frame = self._convert_to_color(frame_data, info)
                
                if rgb_frame is not None:
                    with self.frame_lock:
                        self.frame = rgb_frame
                        if self.frame_count % 30 == 0:
                            print(f"✅ Frame {self.frame_count} convertit cu succes!")
                else:
                    print("⚠️ Conversie eșuată, creez frame placeholder")
                    self._create_placeholder_frame(info)
                        
        except Exception as e:
            print(f"⚠️ Eroare callback frame: {e}")
    
    def _convert_to_color(self, frame_data, info):
        """Convertește frame_data în format RGB cu multiple metode"""
        try:
            # Extrage informațiile din frame info
            frame_info = info.get('raw', {}).get('frame', {})
            format_str = frame_info.get('format', '')
            width = frame_info.get('info', {}).get('width', 640)
            height = frame_info.get('info', {}).get('height', 480)
            
            print(f"📺 Format detectat: {format_str}, Dimensiuni: {width}x{height}")
            
            # Metoda 1: Dacă este deja RGB sau RGBA
            if len(frame_data.shape) == 3:
                if frame_data.shape[2] == 3:
                    print("✅ Frame deja RGB")
                    return frame_data
                elif frame_data.shape[2] == 4:
                    print("✅ Frame RGBA -> RGB")
                    return cv2.cvtColor(frame_data, cv2.COLOR_RGBA2RGB)
            
            # Metoda 2: YUV420 PLANAR (formatul detectat!)
            if 'YUV420' in format_str and 'PLANAR' in format_str:
                print("🎯 Procesare YUV420 PLANAR...")
                return self._convert_yuv420_planar_to_rgb(frame_data, width, height)
            
            # Metoda 3: Frame NV12 (format comun pentru drone)
            if 'NV12' in format_str:
                return self._convert_nv12_to_rgb(frame_data, {'width': width, 'height': height})
            
            # Metoda 4: Frame YUV420 standard
            if len(frame_data.shape) == 1:
                print("🎯 Procesare YUV420 standard...")
                return self._convert_yuv420_to_rgb(frame_data, {'width': width, 'height': height})
            
            # Metoda 5: Frame grayscale la RGB
            if len(frame_data.shape) == 2:
                print("📺 Conversie Grayscale -> RGB")
                return cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
            
            # Metoda 6: Încearcă autodetecție bazată pe dimensiuni
            return self._auto_detect_format(frame_data, {'width': width, 'height': height})
            
        except Exception as e:
            print(f"⚠️ Eroare conversie culori: {e}")
            return None
    
    def _convert_yuv420_planar_to_rgb(self, frame_data, width, height):
        """Convertește YUV420 PLANAR la RGB (formatul specific Anafi)"""
        try:
            # YUV420 PLANAR: Y plane + U plane + V plane separate
            total_pixels = width * height
            y_size = total_pixels
            u_size = total_pixels // 4
            v_size = total_pixels // 4
            
            print(f"📺 YUV420 PLANAR: Y={y_size}, U={u_size}, V={v_size}, Total={frame_data.size}")
            
            # Verifică dacă avem suficiente date
            if frame_data.size < (y_size + u_size + v_size):
                print("⚠️ Nu avem suficiente date pentru YUV420 PLANAR")
                return None
            
            # Extrage planurile
            y_plane = frame_data[:y_size].reshape((height, width))
            u_plane = frame_data[y_size:y_size + u_size].reshape((height // 2, width // 2))
            v_plane = frame_data[y_size + u_size:y_size + u_size + v_size].reshape((height // 2, width // 2))
            
            # Redimensionează U și V la dimensiunea Y
            u_resized = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
            v_resized = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Combină planurile în format YUV
            yuv_image = np.zeros((height, width, 3), dtype=np.uint8)
            yuv_image[:, :, 0] = y_plane  # Y
            yuv_image[:, :, 1] = u_resized  # U
            yuv_image[:, :, 2] = v_resized  # V
            
            # Convertește YUV la RGB
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
            
            print("✅ Conversie YUV420 PLANAR -> RGB reușită!")
            return rgb_image
            
        except Exception as e:
            print(f"⚠️ Eroare conversie YUV420 PLANAR: {e}")
            return None
    
    def _convert_nv12_to_rgb(self, frame_data, info):
        """Convertește NV12 la RGB"""
        try:
            width = info.get('width', 640)
            height = info.get('height', 480)
            
            # NV12: Y plane + UV plane
            y_size = width * height
            y_plane = frame_data[:y_size].reshape((height, width))
            uv_plane = frame_data[y_size:].reshape((height // 2, width // 2, 2))
            
            # Creează YUV frame
            yuv = np.zeros((height * 3 // 2, width), dtype=np.uint8)
            yuv[:height, :] = y_plane
            
            # Interpolează UV
            u = cv2.resize(uv_plane[:, :, 0], (width, height))
            v = cv2.resize(uv_plane[:, :, 1], (width, height))
            
            yuv_frame = np.stack([y_plane, u, v], axis=2)
            rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
            
            print("✅ Conversie NV12 -> RGB reușită")
            return rgb_frame
            
        except Exception as e:
            print(f"⚠️ Eroare conversie NV12: {e}")
            return None
    
    def _convert_yuv420_to_rgb(self, frame_data, info):
        """Convertește YUV420 la RGB"""
        try:
            width = info.get('width', 640)
            height = info.get('height', 480)
            
            # YUV420: Y + U + V planes
            y_size = width * height
            uv_size = y_size // 4
            
            y_plane = frame_data[:y_size].reshape((height, width))
            u_plane = frame_data[y_size:y_size + uv_size].reshape((height // 2, width // 2))
            v_plane = frame_data[y_size + uv_size:y_size + 2 * uv_size].reshape((height // 2, width // 2))
            
            # Redimensionează U și V
            u_resized = cv2.resize(u_plane, (width, height))
            v_resized = cv2.resize(v_plane, (width, height))
            
            # Combină planurile
            yuv_frame = np.stack([y_plane, u_resized, v_resized], axis=2)
            rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
            
            print("✅ Conversie YUV420 -> RGB reușită")
            return rgb_frame
            
        except Exception as e:
            print(f"⚠️ Eroare conversie YUV420: {e}")
            return None
    
    def _auto_detect_format(self, frame_data, info):
        """Încearcă autodetecția formatului"""
        try:
            width = info.get('width', 640)
            height = info.get('height', 480)
            
            # Calculează dimensiunea așteptată pentru diferite formate
            rgb_size = width * height * 3
            yuv_size = width * height * 3 // 2
            
            data_size = frame_data.size
            
            if data_size == rgb_size:
                print("📺 Autodetecție: probabil RGB")
                return frame_data.reshape((height, width, 3))
            elif data_size == yuv_size:
                print("📺 Autodetecție: probabil YUV420")
                return self._convert_yuv420_to_rgb(frame_data.flatten(), info)
            else:
                print(f"📺 Autodetecție: dimensiune nerecunoscută {data_size}")
                # Încearcă să reformeze ca grayscale
                pixels = width * height
                if data_size >= pixels:
                    gray_frame = frame_data.flatten()[:pixels].reshape((height, width))
                    return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            
        except Exception as e:
            print(f"⚠️ Eroare autodetecție: {e}")
            
        return None
    
    def _create_placeholder_frame(self, info):
        """Creează un frame placeholder colorat"""
        width = info.get('width', 640)
        height = info.get('height', 480)
        
        # Creează un frame colorat cu gradient
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient colorat
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                placeholder[y, x] = [r, g, b]
        
        # Adaugă text
        cv2.putText(placeholder, f"Video Stream {width}x{height}", 
                   (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, f"Frame #{self.frame_count}", 
                   (50, height//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        with self.frame_lock:
            self.frame = placeholder
            print(f"✅ Frame placeholder colorat creat: {width}x{height}")
    
    def _h264_frame_callback(self, frame):
        """Callback pentru frame-uri H264 - încearcă decodarea"""
        try:
            if frame is not None and self.video_running:
                print(f"📺 Frame H264 primit: {frame.info()}")
                
                # Aici poți încerca să decodezi H264 dacă ai codec-ul disponibil
                # Pentru simplitate, folosim doar raw frames
                
        except Exception as e:
            print(f"⚠️ Eroare callback H264: {e}")
    
    def _start_callback(self):
        """Callback la pornirea streaming-ului"""
        print("▶️ Streaming color pornit!")
    
    def _end_callback(self):
        """Callback la oprirea streaming-ului"""
        print("⏹️ Streaming color oprit!")
    
    def _flush_callback(self, *args):
        """Callback pentru flush"""
        print("🔄 Flush streaming color")
    
    def get_frame(self):
        """Returnează frame-ul curent"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop_video_stream(self):
        """Oprește streaming-ul video"""
        self.video_running = False
        try:
            self.drone.streaming.stop()
            print("📹 Streaming video color oprit!")
        except Exception as e:
            print(f"⚠️ Eroare la oprirea streaming: {e}")

def enhanced_display_video(video_stream):
    """Afișează video cu îmbunătățiri pentru culori"""
    print("🖥️ Pornire fereastră video îmbunătățită...")
    
    no_frame_count = 0
    max_wait = 150  # 15 secunde
    
    # Creează fereastră (compatibilitate cu versiuni vechi OpenCV)
    try:
        cv2.namedWindow('Parrot Anafi - Color Video', cv2.WINDOW_RESIZABLE)
    except AttributeError:
        cv2.namedWindow('Parrot Anafi - Color Video', cv2.WINDOW_NORMAL)
    
    while no_frame_count < max_wait:
        frame = video_stream.get_frame()
        if frame is not None:
            print("📺 Primul frame primit! Afișez video color...")
            
            while True:
                frame = video_stream.get_frame()
                if frame is not None:
                    # Asigură-te că frame-ul este în format corect
                    if len(frame.shape) == 3:
                        # Convertește de la RGB la BGR pentru OpenCV
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        display_frame = frame
                    
                    # Redimensionează pentru afișare
                    height, width = display_frame.shape[:2]
                    if width > 1000:
                        scale = 1000 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    # Adaugă overlay colorat
                    cv2.putText(display_frame, "Parrot Anafi - Color Stream", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Apasa 'q' pentru iesire", 
                               (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Adaugă un indicator de culoare în colțul din dreapta sus
                    h, w = display_frame.shape[:2]
                    cv2.rectangle(display_frame, (w-60, 10), (w-10, 60), (0, 255, 255), -1)
                    cv2.putText(display_frame, "LIVE", (w-55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    cv2.imshow('Parrot Anafi - Color Video', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Afișează ecran de așteptare colorat
                    wait_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # Fundal gradient
                    for y in range(480):
                        for x in range(640):
                            wait_frame[y, x] = [50, 50 + y//10, 100 + x//10]
                    
                    cv2.putText(wait_frame, "Asteptare frame video color...", 
                               (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow('Parrot Anafi - Color Video', wait_frame)
                    
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
            
            cv2.destroyAllWindows()
            return
        
        no_frame_count += 1
        time.sleep(0.1)
    
    print("⚠️ Nu s-au primit frame-uri video în 15 secunde")
    cv2.destroyAllWindows()

def color_video_test():
    """Test dedicat pentru video color"""
    print("🎨 Test video color - Parrot Anafi")
    
    drone = olympe.Drone(DRONE_IP)
    color_video = ColorVideoStream(drone)
    
    try:
        print("📡 Conectare...")
        drone.connect()
        print("✅ Conectat!")
        
        # Pornește streaming color
        if color_video.start_video_stream():
            print("🎨 Streaming color pornit! Așteaptă 3 secunde...")
            time.sleep(3)
            
            # Afișează video
            enhanced_display_video(color_video)
        else:
            print("❌ Nu s-a putut porni streaming-ul color")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        
    finally:
        print("🔌 Oprire și deconectare...")
        color_video.stop_video_stream()
        drone.disconnect()
        print("👋 Gata!")

def flight_with_color_video():
    """Test complet cu zbor și video color"""
    print("🚁🎨 Test complet - Zbor cu video color")
    
    drone = olympe.Drone(DRONE_IP)
    color_video = ColorVideoStream(drone)
    
    try:
        print("📡 Conectare...")
        drone.connect()
        print("✅ Conectat!")
        
        # Pornește video color
        video_started = color_video.start_video_stream()
        if video_started:
            print("🎨 Video color pornit - așteaptă 3 secunde...")
            time.sleep(3)
            
            # Pornește thread-ul video
            video_thread = threading.Thread(target=enhanced_display_video, args=(color_video,))
            video_thread.daemon = True
            video_thread.start()
        
        print("🛫 Decolare...")
        drone(TakeOff()).wait().success()
        print("✅ Decolat!")
        
        # Zbor cu video color
        print("⏱️ Zbor cu video color - 15 secunde...")
        for i in range(15):
            time.sleep(1)
            print(f"   ⏰ {15-i} secunde rămase...")
        
        print("🛬 Aterizare...")
        drone(Landing()).wait().success()
        print("✅ Aterizat!")
        
        print("🎉 Test complet cu video color reușit!")
        
        if video_started:
            print("🎨 Video color rămâne activ - apasă 'q' pentru închidere")
            video_thread.join()
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        
    finally:
        print("🔌 Oprire și deconectare...")
        color_video.stop_video_stream()
        drone.disconnect()
        print("👋 Gata!")

def original_video_test():
    """Test video original pentru comparație"""
    print("📹 Test video original - Parrot Anafi")
    
    drone = olympe.Drone(DRONE_IP)
    
    class SimpleVideoStream:
        def __init__(self, drone):
            self.drone = drone
            self.video_running = False
            self.frame_count = 0
            
        def start_streaming(self):
            try:
                print("📹 Pornire streaming simplu...")
                self.drone.streaming.start()
                self.video_running = True
                
                self.monitor_thread = threading.Thread(target=self._monitor_stream)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
                
                print("📹 Streaming simplu pornit!")
                return True
                
            except Exception as e:
                print(f"❌ Eroare streaming simplu: {e}")
                return False
        
        def _monitor_stream(self):
            while self.video_running:
                try:
                    if hasattr(self.drone.streaming, 'state'):
                        state = self.drone.streaming.state
                        if state:
                            self.frame_count += 1
                            if self.frame_count % 30 == 0:
                                print(f"📺 Streaming activ - {self.frame_count} frame-uri")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"⚠️ Eroare monitorizare: {e}")
                    time.sleep(1)
        
        def stop_streaming(self):
            self.video_running = False
            try:
                self.drone.streaming.stop()
                print("📹 Streaming simplu oprit!")
            except Exception as e:
                print(f"⚠️ Eroare oprire streaming: {e}")
    
    simple_stream = SimpleVideoStream(drone)
    
    try:
        print("📡 Conectare...")
        drone.connect()
        print("✅ Conectat!")
        
        if simple_stream.start_streaming():
            print("📹 Streaming simplu activ!")
            print("⏱️ Rulează 30 secunde...")
            
            for i in range(30):
                time.sleep(1)
                print(f"   ⏰ {30-i} secunde rămase...")
            
            print("📹 Test video original terminat!")
        else:
            print("❌ Nu s-a putut porni streaming-ul")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        
    finally:
        print("🔌 Oprire și deconectare...")
        simple_stream.stop_streaming()
        drone.disconnect()
        print("👋 Gata!")

if __name__ == "__main__":
    print("=== PARROT ANAFI COLOR VIDEO ===")
    print("Selectează modul de testare:")
    print("1. Test doar video color (fără zbor)")
    print("2. Test complet (zbor + video color)")
    print("3. Test video original (pentru comparație)")
    
    choice = input("Alege (1, 2 sau 3): ").strip()
    
    if choice == "1":
        color_video_test()
    elif choice == "2":
        flight_with_color_video()
    elif choice == "3":
        original_video_test()
    else:
        print("Alegere invalidă. Pornesc testul video color...")
        color_video_test()