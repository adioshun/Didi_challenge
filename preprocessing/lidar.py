import os
os.environ['HOME'] = '/root'

SEED = 202


# std libs
import glob


# num libs
import math
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import cv2
import mayavi.mlab as mlab


from net.utility.draw import *


## save mpg:
##    os.system('ffmpeg -y -loglevel 0 -f image2 -r 15 -i %s/test/predictions/%%06d.png -b:v 2500k %s'%(out_dir,out_avi_file))
##
##----------------------------------------------------------------------------

## preset view points
#  azimuth=180,elevation=0,distance=100,focalpoint=[0,0,0]
## mlab.view(azimuth=azimuth,elevation=elevation,distance=distance,focalpoint=focalpoint)
MM_TOP_VIEW  = 180, 0, 120, [0,0,0]
MM_PER_VIEW1 = 120, 30, 70, [0,0,0]
MM_PER_VIEW2 = 30, 45, 100, [0,0,0]
MM_PER_VIEW3 = 120, 30,100, [0,0,0]



## draw  --------------------------------------------

def draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1):

    pxs=lidar['x']
    pys=lidar['y']
    pzs=lidar['z']
    prs=lidar['intensity']
    #prs=arr['ring']
    prs = np.clip(prs/15,0,1)

    #draw grid
    if is_grid:
        L=25
        dL=5
        Z=-2
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-L,L+1,dL):
            x1,y1,z1 = -L, y, Z
            x2,y2,z2 =  L, y, Z
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.3,0.3,0.3), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-L,L+1,dL):
            x1,y1,z1 = x,-L, Z
            x2,y2,z2 = x, L, Z
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.3,0.3,0.3), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if is_axis:
        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)

        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, line_width=2, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, line_width=2, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, line_width=2, figure=fig)


    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        #colormap='bone',  #(0.7,0.7,0.7),  #'gnuplot',  #'bone',  #'spectral',  #'copper',
        #color=(0.9,0.9,0.9),
        #color=(0.9,0.9,0),
        scale_factor=1,
        figure=fig)

def draw_didi_boxes3d(fig, boxes3d, is_number=False, color=(1,1,1), line_width=1):

    if boxes3d.shape==(8,3): boxes3d=boxes3d.reshape(1,8,3)

    num = len(boxes3d)
    for n in range(num):
        b = boxes3d[n]

        if is_number:
            mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)






# run #################################################################

def mark_gt_box3d( lidar_dir, gt_boxes3d_dir, mark_dir):

    os.makedirs(mark_dir, exist_ok=True)
    fig   = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(500, 500))
    dummy = np.zeros((10,10,3),dtype=np.uint8)

    for file in sorted(glob.glob(lidar_dir + '/*.npy')):
        name = os.path.basename(file).replace('.npy','')

        lidar_file   = lidar_dir     +'/'+name+'.npy'
        boxes3d_file = gt_boxes3d_dir+'/'+name+'.npy'
        lidar   = np.load(lidar_file)
        boxes3d = np.load(boxes3d_file)

        mlab.clf(fig)
        draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1)
        if len(boxes3d)!=0:
            draw_didi_boxes3d(fig, boxes3d)

        azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
        mlab.view(azimuth,elevation,distance,focalpoint)
        mlab.show(1)
        imshow('dummy',dummy)
        cv2.waitKey(1)

        mlab.savefig(mark_dir+'/'+name+'.png',figure=fig)




# main #################################################################
# for demo data:  /root/share/project/didi/data/didi/didi-2/Out/1/15

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    lidar_dir      ='./lidar'
    gt_boxes3d_dir ='./processed/gt_boxes3d'
    mark_dir       ='./processed/mark-gt-box3d'
    avi_file       ='./processed/mark-gt-box3d.avi'

    mark_gt_box3d(lidar_dir,gt_boxes3d_dir,mark_dir)
    dir_to_avi(avi_file, mark_dir)

