import cv2
import numpy as np
from typing import List, Tuple, Dict
import math
import argparse
import json
import os
from datetime import datetime

class AdvancedCircleDetector:
    """Detector avansat pentru cercuri cu configurari personalizabile si urmarire temporala"""
    
    def __init__(self, config_file: str = "circle_config.json"):
        self.config_file = config_file
        self.load_config()
        
        self.cap = None
        self.frame_buffer = []  # Pentru filtrare temporala
        self.previous_circles = []  # Pentru urmarirea cercurilor
        self.max_history = 5  # Numărul de cadre memorate pentru urmarire
        
        self.detection_stats = {
            'total_frames': 0,
            'circles_detected': 0,
            'max_circles_in_frame': 0
        }
        
        # Culori pentru bounding boxes
        self.colors = {
            'closest': (0, 255, 0),      # Verde
            'second': (0, 255, 255),     # Galben
            'third': (255, 165, 0),      # Portocaliu
            'others': (128, 128, 128)    # Gri
        }
    
    def load_config(self):
        """Incarca configuratia din fisier"""
        default_config = {
            "detection": {
                "min_radius": 20,
                "max_radius": 200,
                "min_distance": 50,
                "param1": 50,  # Threshold mai relaxat pentru edge detection
                "param2": 10,  # Threshold mai relaxat pentru centru
                "dp": 1,
                "min_circle_strength": 0.3,
                "edge_density_threshold": 0.2
            },
            "preprocessing": {
                "gaussian_blur": 7,  # Kernel mai mare pentru reducerea zgomotului
                "gaussian_sigma": 1,
                "canny_low": 30,     # Praguri mai relaxate pentru Canny
                "canny_high": 100,
                "contrast_enhancement": True
            },
            "validation": {
                "min_edge_ratio": 0.1,  # Prag mai strict pentru margini
                "max_noise_ratio": 0.15, # Prag mai strict pentru zgomot
                "circularity_threshold": 0.4,
                "min_edge_consistency": 0.2,  # Prag nou pentru consistenta
                "min_contrast_score": 0.15     # Prag nou pentru contrast
            },
            "display": {
                "show_center": True,
                "show_radius": True,
                "show_bbox": True,
                "show_stats": True,
                "show_edges": False,
                "show_validation_info": True
            },
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "tracking": {
                "max_distance": 50,  # Distantă maximă pentru potrivirea cercurilor
                "max_radius_diff": 20,  # Diferență maximă de rază
                "min_tracking_score": 3  # Număr minim de cadre pentru validare
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                self.config = self._merge_configs(default_config, loaded_config)
                print(f"✅ Configuratie incarcata din {self.config_file}")
            else:
                self.config = default_config
                self.save_config()
                print(f"📝 Configuratie default creata: {self.config_file}")
        except Exception as e:
            print(f"⚠️ Eroare la incarcarea configuratiei: {e}")
            self.config = default_config
    
    def _merge_configs(self, default_config, loaded_config):
        """Merge configurația încărcată cu cea default"""
        merged = default_config.copy()
        for section, values in loaded_config.items():
            if section in merged:
                if isinstance(values, dict):
                    merged[section].update(values)
                else:
                    merged[section] = values
            else:
                merged[section] = values
        return merged
    
    def save_config(self):
        """Salveaza configuratia in fisier"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"💾 Configuratie salvata in {self.config_file}")
        except Exception as e:
            print(f"❌ Eroare la salvarea configuratiei: {e}")
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """Initializeaza camera cu setari din config"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"❌ Nu pot deschide camera {camera_id}")
                return False
            
            cam_config = self.config['camera']
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
            self.cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])
            
            print(f"✅ Camera initializata: {cam_config['width']}x{cam_config['height']} @ {cam_config['fps']}fps")
            return True
        except Exception as e:
            print(f"❌ Eroare la initializarea camerei: {e}")
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesseaza frame-ul cu filtrare temporala si edge detection"""
        # Converteste la grayscale si adauga la buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        
        # Aplica filtrare temporala (media ultimelor cadre)
        if len(self.frame_buffer) > 1:
            gray = np.mean(self.frame_buffer, axis=0).astype(np.uint8)
        
        # Aplica blur pentru reducerea zgomotului
        blur_size = self.config['preprocessing']['gaussian_blur']
        sigma = self.config['preprocessing']['gaussian_sigma']
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
        
        # Imbunatateste contrastul
        if self.config['preprocessing']['contrast_enhancement']:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
        else:
            enhanced = blurred
        
        # Edge detection
        edges = cv2.Canny(enhanced, 
                         self.config['preprocessing']['canny_low'], 
                         self.config['preprocessing']['canny_high'])
        
        # Morfologie pentru consolidarea marginilor
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return enhanced, edges
    
    def match_circles(self, new_circles: List[Dict], previous_circles: List[Dict]) -> List[Dict]:
        """Coreleaza cercurile noi cu cele din cadrele anterioare"""
        matched_circles = []
        tracking_config = self.config['tracking']
        
        for new_circle in new_circles:
            x, y = new_circle['center']
            r = new_circle['radius']
            best_match = None
            min_distance = float('inf')
            
            for prev_circle in previous_circles:
                px, py = prev_circle['center']
                pr = prev_circle['radius']
                distance = math.sqrt((x - px)**2 + (y - py)**2)
                radius_diff = abs(r - pr)
                
                if (distance < tracking_config['max_distance'] and 
                    radius_diff < tracking_config['max_radius_diff']):
                    if distance < min_distance:
                        min_distance = distance
                        best_match = prev_circle
            
            if best_match:
                new_circle['tracking_score'] = best_match.get('tracking_score', 0) + 1
            else:
                new_circle['tracking_score'] = 1
            matched_circles.append(new_circle)
        
        return matched_circles
    
    def detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detecteaza cercurile din frame cu validare si urmarire"""
        enhanced, edges = self.preprocess_frame(frame)
        det_config = self.config['detection']
        
        # Detecteaza cercurile folosind HoughCircles
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=det_config['dp'],
            minDist=det_config['min_distance'],
            param1=det_config['param1'],
            param2=det_config['param2'],
            minRadius=det_config['min_radius'],
            maxRadius=det_config['max_radius']
        )
        
        detected_circles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                circle_metrics = self._validate_circle(x, y, r, edges, frame)
                
                if circle_metrics['is_valid']:
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2
                    distance_from_center = math.sqrt(
                        (x - frame_center_x) ** 2 + (y - frame_center_y) ** 2
                    )
                    
                    detected_circles.append({
                        'center': (x, y),
                        'radius': r,
                        'area': math.pi * r * r,
                        'bbox': (x - r, y - r, 2 * r, 2 * r),
                        'distance_from_center': distance_from_center,
                        'visibility_score': self._calculate_visibility(x, y, r, frame.shape),
                        'validation_metrics': circle_metrics,
                        'quality_score': circle_metrics['overall_score'],
                        'tracking_score': 1  # Initializat, va fi actualizat in match_circles
                    })
        
        # Coreleaza cu cercurile anterioare
        detected_circles = self.match_circles(detected_circles, self.previous_circles)
        self.previous_circles = detected_circles[-self.max_history:]
        
        # Filtreaza cercurile cu tracking_score insuficient
        return [c for c in detected_circles if c['tracking_score'] >= self.config['tracking']['min_tracking_score']]
    
    def _validate_circle(self, x: int, y: int, r: int, edges: np.ndarray, frame: np.ndarray) -> Dict:
        """Valideaza cercul cu praguri individuale stricte"""
        validation_config = self.config['validation']
        
        # Genereaza puncte pe circumferinta
        angles = np.linspace(0, 2*np.pi, max(50, int(2*np.pi*r/3)))
        circle_points = []
        
        for angle in angles:
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                circle_points.append((px, py))
        
        if len(circle_points) < 10:
            return {'is_valid': False, 'reason': 'insufficient_points', 'overall_score': 0.0}
        
        # 1. Verifica densitatea marginilor
        edge_count = sum(1 for px, py in circle_points if edges[py, px] > 0)
        total_points = len(circle_points)
        edge_ratio = edge_count / total_points
        
        # 2. Verifica consistenta marginilor
        edge_consistency = self._check_edge_consistency(circle_points, edges)
        
        # 3. Verifica contrastul
        contrast_score = self._calculate_contrast_score(x, y, r, frame)
        
        # 4. Verifica circularitatea
        circularity_score = self._calculate_circularity(circle_points, edges)
        
        # Calculeaza scorul final cu ponderi ajustate
        scores = {
            'edge_ratio': edge_ratio,
            'edge_consistency': edge_consistency,
            'contrast_score': contrast_score,
            'circularity_score': circularity_score
        }
        
        overall_score = (
            edge_ratio * 0.30 +        # Redus de la 35%
            edge_consistency * 0.30 +  # Crescut de la 25%
            contrast_score * 0.25 +
            circularity_score * 0.15
        )
        
        # Validare cu praguri individuale
        is_valid = (
            edge_ratio >= validation_config['min_edge_ratio'] and
            edge_consistency >= validation_config['min_edge_consistency'] and
            contrast_score >= validation_config['min_contrast_score'] and
            circularity_score >= validation_config['circularity_threshold'] and
            overall_score >= self.config['detection']['min_circle_strength']
        )
        
        return {
            'is_valid': is_valid,
            'overall_score': overall_score,
            'edge_ratio': edge_ratio,
            'edge_consistency': edge_consistency,
            'contrast_score': contrast_score,
            'circularity_score': circularity_score,
            'total_edge_points': edge_count,
            'total_points': total_points
        }
    
    def _check_edge_consistency(self, circle_points: List[Tuple], edges: np.ndarray) -> float:
        """Verifica consistenta marginilor"""
        if len(circle_points) < 10:
            return 0.0
        
        edge_segments = []
        current_segment = 0
        
        for px, py in circle_points:
            if edges[py, px] > 0:
                current_segment += 1
            else:
                if current_segment > 0:
                    edge_segments.append(current_segment)
                    current_segment = 0
        
        if current_segment > 0:
            edge_segments.append(current_segment)
        
        if not edge_segments:
            return 0.0
        
        max_segment = max(edge_segments)
        avg_segment = sum(edge_segments) / len(edge_segments)
        
        consistency = min(1.0, (max_segment / len(circle_points)) + (avg_segment / max_segment))
        return consistency
    
    def _calculate_contrast_score(self, x: int, y: int, r: int, frame: np.ndarray) -> float:
        """Calculeaza scorul de contrast"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask_inner = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask_inner, (x, y), max(1, r-5), 255, -1)
        
        mask_outer = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask_outer, (x, y), r+5, 255, -1)
        cv2.circle(mask_outer, (x, y), r, 0, -1)
        
        inner_vals = gray[mask_inner > 0]
        outer_vals = gray[mask_outer > 0]
        
        if len(inner_vals) == 0 or len(outer_vals) == 0:
            return 0.0
        
        inner_mean = np.mean(inner_vals)
        outer_mean = np.mean(outer_vals)
        
        contrast = abs(inner_mean - outer_mean) / 255.0
        return min(1.0, contrast * 2)
    
    def _calculate_circularity(self, circle_points: List[Tuple], edges: np.ndarray) -> float:
        """Calculeaza circularitatea"""
        if len(circle_points) < 10:
            return 0.0
        
        edge_points = [[px, py] for px, py in circle_points if edges[py, px] > 0]
        
        if len(edge_points) < 8:
            return 0.0
        
        edge_points = np.array(edge_points, dtype=np.float32)
        
        try:
            (center_x, center_y), radius = cv2.minEnclosingCircle(edge_points)
            distances = [math.sqrt((px - center_x)**2 + (py - center_y)**2) for px, py in edge_points]
            std_dev = np.std(distances)
            max_allowed_dev = radius * 0.15
            circularity = max(0.0, 1.0 - (std_dev / max_allowed_dev))
            return circularity
        except:
            return 0.0
    
    def _calculate_visibility(self, x: int, y: int, r: int, frame_shape: Tuple) -> float:
        """Calculeaza scorul de vizibilitate"""
        height, width = frame_shape[:2]
        
        left = max(0, x - r)
        right = min(width, x + r)
        top = max(0, y - r)
        bottom = min(height, y + r)
        
        visible_area = (right - left) * (bottom - top)
        total_area = (2 * r) * (2 * r)
        
        return visible_area / total_area if total_area > 0 else 0
    
    def sort_circles_by_proximity(self, circles: List[Dict]) -> List[Dict]:
        """Sorteaza cercurile dupa proximitate si calitate"""
        if not circles:
            return circles
        
        for circle in circles:
            size_score = circle['radius'] / 200.0
            visibility_score = circle['visibility_score']
            center_score = 1.0 / (1.0 + circle['distance_from_center'] / 200.0)
            quality_score = circle['quality_score']
            tracking_score = circle['tracking_score'] / 10.0  # Normalizeaza
            
            circle['proximity_score'] = (
                size_score * 0.20 +
                visibility_score * 0.15 +
                center_score * 0.10 +
                quality_score * 0.45 +
                tracking_score * 0.10  # Adauga influenta urmaririi
            )
        
        return sorted(circles, key=lambda c: c['proximity_score'], reverse=True)
    
    def draw_detections(self, frame: np.ndarray, circles: List[Dict]) -> np.ndarray:
        """Deseneaza detectiile pe frame"""
        result_frame = frame.copy()
        display_config = self.config['display']
        
        if not circles:
            return result_frame
        
        for i, circle in enumerate(circles):
            x, y = circle['center']
            r = circle['radius']
            
            if i == 0:
                color = self.colors['closest']
                label = "CLOSEST"
            elif i == 1:
                color = self.colors['second']
                label = "SECOND"
            elif i == 2:
                color = self.colors['third']
                label = "THIRD"
            else:
                color = self.colors['others']
                label = f"RING {i+1}"
            
            cv2.circle(result_frame, (x, y), r, color, 2)
            
            if display_config['show_center']:
                cv2.circle(result_frame, (x, y), 3, color, -1)
            
            if display_config['show_bbox']:
                bbox_x, bbox_y, bbox_w, bbox_h = circle['bbox']
                cv2.rectangle(result_frame, 
                             (bbox_x, bbox_y), 
                             (bbox_x + bbox_w, bbox_y + bbox_h), 
                             color, 2)
            
            info_lines = [label]
            if display_config['show_radius']:
                info_lines.append(f"R:{r}")
            
            if display_config['show_validation_info']:
                metrics = circle['validation_metrics']
                info_lines.append(f"Q:{circle['quality_score']:.2f}")
                info_lines.append(f"E:{metrics['edge_ratio']:.2f}")
                info_lines.append(f"C:{metrics['edge_consistency']:.2f}")
                info_lines.append(f"CS:{metrics['contrast_score']:.2f}")
                info_lines.append(f"Cir:{metrics['circularity_score']:.2f}")
                info_lines.append(f"T:{circle['tracking_score']}")
            else:
                info_lines.append(f"Score:{circle['proximity_score']:.2f}")
            
            for j, line in enumerate(info_lines):
                cv2.putText(result_frame, line, 
                           (x - r, y - r - 10 - (j * 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, color, 1)
        
        return result_frame
    
    def draw_stats(self, frame: np.ndarray, circles: List[Dict]) -> np.ndarray:
        """Deseneaza statistici pe frame"""
        if not self.config['display']['show_stats']:
            return frame
        
        stats_lines = [
            f"Frame: {self.detection_stats['total_frames']}",
            f"Circles: {len(circles)}",
            f"Max detected: {self.detection_stats['max_circles_in_frame']}",
            f"Total detected: {self.detection_stats['circles_detected']}"
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (15, 30 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_detection(self, camera_id: int = 0, save_frames: bool = False):
        """Ruleaza detectia cu optiuni avansate"""
        if not self.initialize_camera(camera_id):
            return
        
        print("🎯 Detectie cercuri avansata cu validare stricta si urmarire")
        print("===============================================")
        print("📖 Controale:")
        print("   - ESC sau 'q': Iesire")
        print("   - 's': Salveaza frame")
        print("   - 'c': Deschide configurator")
        print("   - 'r': Reset statistici")
        print("   - 'e': Toggle edge view")
        print("   - 'v': Toggle validation info")
        print("   - SPACE: Pausa/Resume")
        
        paused = False
        save_counter = 0
        show_edges = False
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("❌ Nu pot citi frame-ul")
                        break
                    
                    self.detection_stats['total_frames'] += 1
                    
                    circles = self.detect_circles(frame)
                    
                    if circles:
                        self.detection_stats['circles_detected'] += len(circles)
                        self.detection_stats['max_circles_in_frame'] = max(
                            self.detection_stats['max_circles_in_frame'], 
                            len(circles)
                        )
                    
                    sorted_circles = self.sort_circles_by_proximity(circles)
                    result_frame = self.draw_detections(frame, sorted_circles)
                    result_frame = self.draw_stats(result_frame, circles)
                    
                    if show_edges and circles:
                        _, edges = self.preprocess_frame(frame)
                        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                        combined = np.hstack([result_frame, edges_colored])
                        cv2.imshow('Advanced Circle Detection + Edges', combined)
                    else:
                        cv2.imshow('Advanced Circle Detection', result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"circle_detection_{timestamp}_{save_counter:03d}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"💾 Frame salvat: {filename}")
                    save_counter += 1
                elif key == ord('c'):
                    self.interactive_config()
                elif key == ord('r'):
                    self.detection_stats = {'total_frames': 0, 'circles_detected': 0, 'max_circles_in_frame': 0}
                    self.previous_circles = []
                    print("🔄 Statistici resetate")
                elif key == ord('e'):
                    show_edges = not show_edges
                    status = "ON" if show_edges else "OFF"
                    print(f"🔍 Edge view: {status}")
                elif key == ord('v'):
                    self.config['display']['show_validation_info'] = not self.config['display']['show_validation_info']
                    status = "ON" if self.config['display']['show_validation_info'] else "OFF"
                    print(f"📊 Validation info: {status}")
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"⏸️ {status}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Detectie oprita de utilizator")
        
        finally:
            self.cleanup()
    
    def interactive_config(self):
        """Configurator interactiv pentru parametri"""
        print("\n🔧 Configurator parametri")
        print("========================")
        
        try:
            det_config = self.config['detection']
            val_config = self.config['validation']
            track_config = self.config['tracking']
            
            new_min_radius = input(f"Min radius ({det_config['min_radius']}): ").strip()
            if new_min_radius:
                det_config['min_radius'] = int(new_min_radius)
            
            new_max_radius = input(f"Max radius ({det_config['max_radius']}): ").strip()
            if new_max_radius:
                det_config['max_radius'] = int(new_max_radius)
            
            new_param1 = input(f"Param1 ({det_config['param1']}): ").strip()
            if new_param1:
                det_config['param1'] = int(new_param1)
            
            new_param2 = input(f"Param2 ({det_config['param2']}): ").strip()
            if new_param2:
                det_config['param2'] = int(new_param2)
            
            new_min_edge_ratio = input(f"Min edge ratio ({val_config['min_edge_ratio']}): ").strip()
            if new_min_edge_ratio:
                val_config['min_edge_ratio'] = float(new_min_edge_ratio)
            
            new_min_edge_consistency = input(f"Min edge consistency ({val_config['min_edge_consistency']}): ").strip()
            if new_min_edge_consistency:
                val_config['min_edge_consistency'] = float(new_min_edge_consistency)
            
            new_min_contrast_score = input(f"Min contrast score ({val_config['min_contrast_score']}): ").strip()
            if new_min_contrast_score:
                val_config['min_contrast_score'] = float(new_min_contrast_score)
            
            new_min_tracking_score = input(f"Min tracking score ({track_config['min_tracking_score']}): ").strip()
            if new_min_tracking_score:
                track_config['min_tracking_score'] = int(new_min_tracking_score)
            
            self.save_config()
            print("✅ Configuratie actualizata!")
            
        except ValueError as e:
            print(f"❌ Valoare invalida: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ Configurare anulata")
    
    def cleanup(self):
        """Curata resursele"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Resurse curatate")
        
        print("\n📊 Statistici finale:")
        print(f"   - Total frame-uri: {self.detection_stats['total_frames']}")
        print(f"   - Total cercuri detectate: {self.detection_stats['circles_detected']}")
        print(f"   - Max cercuri intr-un frame: {self.detection_stats['max_circles_in_frame']}")

def main():
    """Functia principala cu argumente din linia de comanda"""
    parser = argparse.ArgumentParser(description='Advanced Circle Detector')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--config', type=str, default='circle_config.json', help='Config file path')
    parser.add_argument('--save-frames', action='store_true', help='Enable frame saving')
    
    args = parser.parse_args()
    
    print("🎯 Sistem avansat de detectare cercuri")
    print("======================================")
    
    detector = AdvancedCircleDetector(args.config)
    
    try:
        detector.run_detection(args.camera, args.save_frames)
    except Exception as e:
        print(f"❌ Eroare: {e}")
        import traceback
        print("Traceback complet:")
        traceback.print_exc()
    
    print("👋 Program terminat")

if __name__ == "__main__":
    main()