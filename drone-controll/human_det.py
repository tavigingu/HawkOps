# YOLOV8 LIVE DETECTION CU CAMERA LAPTOP - STATISTICI OPTIMIZATE
import cv2
import numpy as np
from ultralytics import YOLO
import time

class LivePersonDetector:
    def __init__(self, model_path, conf_threshold=0.3):
        """
        Inițializează detectorul live
        
        Args:
            model_path: calea către modelul .pt antrenat
            conf_threshold: pragul de confidence pentru detecții
        """
        print("🚀 Inițializez detectorul live...")
        
        # Încarcă modelul YOLOv8 antrenat
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model încărcat: {model_path}")
        except Exception as e:
            print(f"❌ Eroare la încărcarea modelului: {e}")
            return
        
        self.conf_threshold = conf_threshold
        self.class_names = ['fallen', 'sit', 'standing']  # Clasele tale
        
        # Culori pentru bounding boxes (BGR format pentru OpenCV)
        self.colors = {
            'fallen': (0, 0, 255),    # Roșu - pericol
            'sit': (0, 255, 255),     # Galben - atenție  
            'standing': (0, 255, 0)   # Verde - normal
        }
        
        # Statistici
        self.fps_counter = 0
        self.start_time = time.time()
        self.frame_count = 0
        
    def draw_detection(self, frame, box, class_id, confidence):
        """
        Desenează bounding box și label pe frame
        """
        # Coordonatele bounding box-ului
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Numele clasei și culoarea
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'Class_{class_id}'
        color = self.colors.get(class_name, (255, 255, 255))
        
        # Desenează bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Pregătește textul
        label = f'{class_name}: {confidence:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Desenează background pentru text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Desenează textul
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_stats_compact(self, frame, detections_count, fps):
        """
        Desenează statistici compacte pe frame (versiune mai mică)
        """
        height, width = frame.shape[:2]
        
        # Dimensiuni mai mici pentru cadrul de statistici
        stats_width = 180
        stats_height = 60
        margin = 8
        
        # Background semi-transparent pentru statistici
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin), (margin + stats_width, margin + stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border subțire
        cv2.rectangle(frame, (margin, margin), (margin + stats_width, margin + stats_height), (100, 100, 100), 1)
        
        # Text mai mic și mai compact
        font_scale = 0.4
        font_thickness = 1
        line_height = 18
        
        # FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (margin + 5, margin + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        # Detecții
        cv2.putText(frame, f'Det: {detections_count}', (margin + 5, margin + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        # Confidence
        cv2.putText(frame, f'Conf: {self.conf_threshold:.1f}', (margin + 5, margin + 49), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        return frame
    
    def draw_stats_minimal(self, frame, detections_count, fps):
        """
        Desenează statistici minimale (doar text, fără background)
        """
        # Text foarte simplu în colțul din stânga sus
        text = f'FPS:{fps:.0f} Det:{detections_count} Conf:{self.conf_threshold:.1f}'
        
        # Contur negru pentru text (pentru vizibilitate)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # Textul propriu-zis în alb
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_stats_corner(self, frame, detections_count, fps):
        """
        Desenează statistici în colțul din dreapta sus
        """
        height, width = frame.shape[:2]
        
        # Calculează dimensiunile textului pentru poziționare
        text_fps = f'FPS: {fps:.1f}'
        text_det = f'Det: {detections_count}'
        text_conf = f'Conf: {self.conf_threshold:.1f}'
        
        font_scale = 0.5
        font_thickness = 1
        
        # Poziționare în colțul din dreapta sus
        x_pos = width - 120
        y_start = 20
        line_height = 20
        
        # Background semi-transparent mic
        cv2.rectangle(frame, (x_pos - 5, y_start - 15), (width - 5, y_start + 45), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_pos - 5, y_start - 15), (width - 5, y_start + 45), (100, 100, 100), 1)
        
        # Desenează textele
        cv2.putText(frame, text_fps, (x_pos, y_start), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(frame, text_det, (x_pos, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(frame, text_conf, (x_pos, y_start + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        return frame
    
    def draw_instructions(self, frame):
        """
        Desenează instrucțiunile în partea de jos
        """
        height, width = frame.shape[:2]
        cv2.putText(frame, 'Q:quit  +/-:confidence  R:reset  S:switch stats', 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame
    
    def process_frame(self, frame):
        """
        Procesează un frame și returnează frame-ul cu detecțiile
        """
        # Rulează detecția
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections_count = 0
        
        # Procesează rezultatele
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            detections_count = len(boxes)
            
            # Desenează fiecare detecție
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                frame = self.draw_detection(frame, box, class_id, confidence)
        
        return frame, detections_count
    
    def start_detection(self, camera_id=0, window_size=(1280, 720), stats_mode='compact'):
        """
        Pornește detecția live
        
        Args:
            camera_id: ID-ul camerei (0 pentru camera principală)
            window_size: dimensiunea ferestrei de afișare
            stats_mode: modul de afișare statistici ('compact', 'minimal', 'corner', 'none')
        """
        print(f"📹 Încerc să conectez camera {camera_id}...")
        
        # Inițializează camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Nu pot accesa camera!")
            print("💡 Încearcă să schimbi camera_id (0, 1, 2...)")
            return
        
        # Setează rezoluția camerei
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera conectată cu succes!")
        print("\n🎮 COMENZI:")
        print("  Q - Oprește detecția")
        print("  + - Crește confidence threshold")
        print("  - - Scade confidence threshold")
        print("  R - Reset statistici")
        print("  S - Schimbă modul de afișare statistici")
        print("\n🚀 Începe detecția live...")
        
        # Modurile de afișare statistici
        stats_modes = ['compact', 'minimal', 'corner', 'none']
        current_stats_mode = stats_mode
        
        try:
            while True:
                # Citește frame-ul
                ret, frame = cap.read()
                if not ret:
                    print("❌ Nu pot citi de la cameră!")
                    break
                
                # Procesează frame-ul
                start_process = time.time()
                processed_frame, detections_count = self.process_frame(frame)
                process_time = time.time() - start_process
                
                # Calculează FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Desenează statistici în funcție de mod
                if current_stats_mode == 'compact':
                    processed_frame = self.draw_stats_compact(processed_frame, detections_count, fps)
                elif current_stats_mode == 'minimal':
                    processed_frame = self.draw_stats_minimal(processed_frame, detections_count, fps)
                elif current_stats_mode == 'corner':
                    processed_frame = self.draw_stats_corner(processed_frame, detections_count, fps)
                # 'none' - nu desenează statistici
                
                # Desenează instrucțiunile
                processed_frame = self.draw_instructions(processed_frame)
                
                # Redimensionează pentru afișare
                processed_frame = cv2.resize(processed_frame, window_size)
                
                # Afișează frame-ul
                cv2.imshow('YOLOv8 Live Person Detection', processed_frame)
                
                # Verifică taste apăsate
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("🛑 Opresc detecția...")
                    break
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(0.9, self.conf_threshold + 0.1)
                    print(f"📈 Confidence threshold: {self.conf_threshold:.1f}")
                elif key == ord('-') or key == ord('_'):
                    self.conf_threshold = max(0.1, self.conf_threshold - 0.1)
                    print(f"📉 Confidence threshold: {self.conf_threshold:.1f}")
                elif key == ord('r') or key == ord('R'):
                    self.frame_count = 0
                    self.start_time = time.time()
                    print("🔄 Statistici resetate")
                elif key == ord('s') or key == ord('S'):
                    # Schimbă modul de statistici
                    current_index = stats_modes.index(current_stats_mode)
                    current_stats_mode = stats_modes[(current_index + 1) % len(stats_modes)]
                    print(f"📊 Mod statistici: {current_stats_mode}")
                
        except KeyboardInterrupt:
            print("\n🛑 Întrerupt de utilizator")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera închisă și resurse eliberate")
            
            # Statistici finale
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            print(f"\n📊 STATISTICI FINALE:")
            print(f"   • Total frame-uri: {self.frame_count}")
            print(f"   • Timp total: {total_time:.1f}s")
            print(f"   • FPS mediu: {avg_fps:.1f}")

def main():
    """Funcția principală pentru a rula detectorul live"""
    
    # Calea către modelul tău antrenat
    MODEL_PATH = "best.pt"  # Schimbă cu calea ta
    
    # Verifică dacă modelul există
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelul nu există la: {MODEL_PATH}")
        print("💡 Asigură-te că ai copiat fișierul best.pt în directorul curent")
        print("💡 Sau schimbă MODEL_PATH cu calea corectă")
        return
    
    # Creează detectorul
    detector = LivePersonDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.3  # Ajustează după nevoie
    )
    
    # Pornește detecția live
    # Opțiuni pentru stats_mode: 'compact', 'minimal', 'corner', 'none'
    detector.start_detection(
        camera_id=0,  # Schimbă cu 1, 2, etc. dacă nu funcționează
        window_size=(1280, 720),  # Dimensiunea ferestrei
        stats_mode='compact'  # Schimbă cu 'minimal', 'corner', sau 'none'
    )

if __name__ == "__main__":
    main()