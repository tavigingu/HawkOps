FROM osrf/ros:noetic-desktop-full

# Setează variabilele de mediu
ENV DEBIAN_FRONTEND=noninteractive

# Instalează toate dependențele necesare
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3-pip \
    python3-catkin-tools \
    libeigen3-dev \
    libopencv-dev \
    libceres-dev \
    ros-noetic-cv-bridge \
    ros-noetic-tf \
    ros-noetic-message-filters \
    ros-noetic-image-transport \
    ros-noetic-rosbridge-server \
    ros-noetic-web-video-server \
    ros-noetic-tf2-web-republisher \
    && rm -rf /var/lib/apt/lists/*

# Creează workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Inițializează workspace-ul gol
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Setup environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

# Instalează dependențe Python
RUN pip3 install numpy opencv-python

WORKDIR /catkin_ws

# Expune porturile
EXPOSE 9090 8080

CMD ["/bin/bash"]
