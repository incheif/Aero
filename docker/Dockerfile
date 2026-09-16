# syntax=docker/dockerfile:1
FROM osrf/ros:humble-desktop-full

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies & TurtleBot3 simulation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-msgs \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-plugins \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    procps \
    psmisc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set standard environment variables for deterministic headless execution
ENV TURTLEBOT3_MODEL=waffle
ENV GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:$GAZEBO_MODEL_PATH
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_QPA_PLATFORM=offscreen
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# Working directory
WORKDIR /workspace

# Install Python requirements
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Source ROS 2 environment in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
