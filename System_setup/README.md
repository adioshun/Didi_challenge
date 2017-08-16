||내용|진행상황|참고|
|-|-|-|-|
|1|ROS설치|완료|ubuntu 14.04|
|2|시각화|완료|.|
|3|extracting data from rosbag|완료|python2|
|4|converting lidar dumy to projection files|진행중|python2|
|5|train a simple CNN|진행중|python3|

## 1. Install ROS Kinetic (Above)

- os: ubuntu 14.04 (google Cloud)
- python : miniconda2(python 2)

## ROS 설치 
> [참고](https://github.com/adioshun/Blog_Jekyll/blob/master/2017-05-18-ROS.md)

## 2. 시각화 
### Velodyne 
- 설치 : ` apt-get install ros-indigo-velodyne`
- 실행 : ` roslaunch velodyne_pointcloud 32e_points.launch` #terminal 1

## Install RViz viewer
- 설치 : Create a ROS-Didi Challenge Workspace(omgteam)

```bash
git clone https://github.com/omgteam/Didi-competition-solution.git
cd Didi-competition-solution
catkin_make
echo "source ~/Didi-competition-solution/devel/setup.bash" >> ~/.bashrc
source ~/Didi-competition-solution/devel/setup.bash
```
- 실행
```bash
roslaunch didi_visualize display_rosbag_rviz.launch rosbag_file:=/root/data/15.bag #terminal 2
```
> 출처 : https://github.com/omgteam/Didi-competition-solution
> - [Jokla코드활용](https://github.com/jokla/didi_challenge_ros)

## 3. extracting data from rosbag
images to png files and lidar point clouds to numpy npy files

code : [run_dump_lidar.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-
04/didi_data/ros_scripts/run_dump_lidar.py)

- 실행 
```bash 
conda env create --file python2.yml --name python2
source activate python2
python run_dump_lidar.py # terminal 3
```


## 4. converting lidar dumy to projection files

code : [lidar.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/lidar.py),  [lidar_top.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/lidar_top.py), [lidar_surround.py](https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/lidar_surround.py)

- 실행 
```bash 
conda env create --file python3.yml --name python3
source activate python3
python lidar.py # terminal 3
```


에러: 
- No module names mayavi.mlab
  - python2 : conda install -c anaconda mayavi 
  - python3 : conda install -c clinicalgraphics vtk=7.1.0; pip install mayavi 
- No module named builtins -> pip install future
- No module named easydict-> sudo pip install easydict; pip install easydict 
- No module named simplejson -> pip install simplejson
- libcudart.so.8.0: cannot open shared object file -> [Faster R-CNN 클론 후 make](https://github.com/smallcorgi/Faster-RCNN_TF#installation-sufficient-for-the-demo)
- No module named cython_box -> [임시] net/processing/boxes.py의 from net.processing.cython_box import box_overlaps주석처리 
- No module named cpu_nms -> cpu_nms.pyx -> import pyximport; pyximport.install()
- conda install pyqt=4
- API 'QDate' has already been set to version 1 -> [임시] kernel restart -> import mayavi.mlab as mlab
- ImportError: Could not import backend for traits -> conda install -c conda-forge pyside=1.2.4 ,[참고](https://github.com/enthought/mayavi/issues/483)

## 5. train a simple CNN


---
> https://github.com/yukitsuji/ubuntu_setup
