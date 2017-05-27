||내용|진행상황|참고|
|-|-|-|-|
|1|ROS설치|완료|Docker|
|2|시각화|완료|RViz|
|3|extracting data from rosbag|완료|run_dump_lidar.py|
|4|converting lidar dumy to projection files|완료|lidar*.py|
|5|train a simple CNN|진행중|train.py|


## 1. Install ROS Kinetic (Above)

### Docker 

1. docker pull adioshun/ros:full-python2
2. mkdir ~/docker_share
3. xhost +
3. docker run -i -t --name ROS --volume /home/adioshun/docker_share:/root/share 4621d4fe2959 /bin/bash
4. docker start ROS;docker attach ROS

실행 : `docker run -it -e NO_AT_BRIDGE=1 -e QT_X11_NO_MITSHM=1 -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/.docker.xauth:/tmp/.docker.xauth -e XAUTHORITY=/tmp/.docker.xauth -p 8586:8586 -v /home/adioshun/docker_share:/root/share --name ROS adioshun/ros:full /bin/bash`

### Install Velodyne drivers
설치 
```bash
apt-get install ros-indigo-velodyne
```

실행 
```bash
roslaunch velodyne_pointcloud 32e_points.launch #terminal 1
```

## 2. Install RViz viewer
설치 : Create a ROS-Didi Challenge Workspace(omgteam)

```bash
git clone https://github.com/omgteam/Didi-competition-solution.git
cd Didi-competition-solution
catkin_make
echo "source ~/Didi-competition-solution/devel/setup.bash" >> ~/.bashrc
source ~/Didi-competition-solution/devel/setup.bash
```
실행
```bash
roslaunch didi_visualize display_rosbag_rviz.launch rosbag_file:=/root/data/15.bag
```

> 출처 : https://github.com/omgteam/Didi-competition-solution
> - [Jokla코드활용](https://github.com/jokla/didi_challenge_ros)

## 3. extracting data from rosbag
images to png files and lidar point clouds to numpy npy files

code : [run_dump_lidar.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-
04/didi_data/ros_scripts/run_dump_lidar.py)

실행 : python run_dump_lidar.py

## 4. converting lidar dumy to projection files

code : [lidar.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/lidar.py),  [lidar_top.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/lidar_top.py), [lidar_surround.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/lidar_surround.py)

에러: No module names mayavi.mlab -
- python2 : conda install -c anaconda mayavi 
- python3 : conda install -c clinicalgraphics vtk=7.1.0; pip install mayavi 
```
# No module named builtins -> pip install future
# No module named easydict-> sudo pip install easydict; pip install easydict 
# No module named simplejson -> pip install simplejson
# No module named cython_box -> [임시] net/processing/boxes.py의 from net.processing.cython_box import box_overlaps주석처리 
# No module named cpu_nms -> cpu_nms.pyx -> import pyximport; pyximport.install()
# conda install pyqt=4
# API 'QDate' has already been set to version 1 -> [임시] kernel restart -> import mayavi.mlab as mlab
```

## 5. train a simple CNN
