#!/bin/bash

# Script pentru configurarea Docker-ului cu GUI și integrarea Anafi

echo "🔧 Configurare Docker pentru ORB-SLAM3 cu GUI..."

# 1. Configurare X11 pentru GUI
echo "📺 Configurare X11..."
xhost +local:docker

# 2. Oprește container-ul existent dacă rulează
echo "🛑 Opresc container-ul existent..."
sudo docker stop orbslam_container 2>/dev/null || true
sudo docker rm orbslam_container 2>/dev/null || true

# 3. Creează directoare pentru partajare
echo "📁 Creez directoare..."
mkdir -p ~/orbslam_data/frames
mkdir -p ~/orbslam_data/results
mkdir -p ~/orbslam_data/config

# 4. Pornește container-ul cu configurări complete
echo "🚀 Pornesc container-ul ORB-SLAM3..."
sudo docker run -d \
    --name orbslam_container \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/orbslam_data:/shared_data:rw \
    --net=host \
    --device=/dev/video0:/dev/video0 \
    tingoose/orb-slam3:latest \
    tail -f /dev/null

echo "✅ Container pornit cu numele 'orbslam_container'"

# 5. Instalează dependențe pentru Olympe SDK
echo "📦 Instalez Olympe SDK pentru Parrot Anafi..."

# Creează mediul virtual pentru Olympe
python3 -m venv ~/anafi_env
source ~/anafi_env/bin/activate

# Instalează dependențele
pip install --upgrade pip
pip install parrot-olympe
pip install opencv-python
pip install numpy

echo "✅ Olympe SDK instalat în ~/anafi_env"

# 6. Creează scriptul de pornire
cat > ~/start_anafi_orbslam.sh << 'EOF'
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
EOF

chmod +x ~/start_anafi_orbslam.sh

# 7. Configurare fișier YAML pentru Anafi
echo "⚙️ Creez configurare pentru camera Anafi..."
cat > ~/orbslam_data/config/Anafi.yaml << 'EOF'
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters pentru Parrot Anafi
#--------------------------------------------------------------------------------------------

# Camera calibration și distortion parameters (aproximative pentru Anafi)
Camera.fx: 1382.58
Camera.fy: 1383.74
Camera.cx: 960.0
Camera.cy: 540.0

Camera.k1: -0.159
Camera.k2: 0.026
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 1920
Camera.height: 1080

# Camera frames per second
Camera.fps: 30.0

# IR projector baseline times fx (aprox)
Camera.bf: 40.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 40.0

# Deptmap values factor
DepthMapFactor: 1.0

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
EOF

echo "✅ Configurare completă!"
echo ""
echo "📋 Pași următori:"
echo "1. Conectează-te la WiFi-ul dronei Anafi"
echo "2. Rulează: ~/start_anafi_orbslam.sh"
echo "3. Folosește comenzile din aplicație pentru control"
echo ""
echo "📁 Fișiere importante:"
echo "  - Script pornire: ~/start_anafi_orbslam.sh"
echo "  - Configurare Anafi: ~/orbslam_data/config/Anafi.yaml"
echo "  - Date partajate: ~/orbslam_data/"
echo "  - Mediu virtual: ~/anafi_env/"