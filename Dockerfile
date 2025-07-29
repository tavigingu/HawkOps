FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libeigen3-dev libopencv-dev \
    libssl-dev libgl1-mesa-dev libglew-dev \
    ffmpeg libavcodec-dev libavutil-dev \
    libavformat-dev libswscale-dev \
    libavdevice-dev libdc1394-22-dev \
    libraw1394-dev libjpeg-dev libpng-dev \
    libtiff5-dev libopenexr-dev \
    x11-apps mesa-utils libepoxy-dev \
    python3-dev python3-numpy

# Pangolin
COPY Pangolin /tmp/Pangolin
WORKDIR /tmp/Pangolin
RUN rm -rf build && mkdir build && cd build && \
    cmake .. && make -j4 && make install && ldconfig

# ORB-SLAM3
COPY ORB_SLAM3_new /tmp/ORB_SLAM3
WORKDIR /tmp/ORB_SLAM3
RUN rm -rf build Thirdparty/*/build && \
    chmod +x build.sh && ./build.sh

WORKDIR /tmp/ORB_SLAM3
CMD ["/bin/bash"]
