from net.utility.file import *
from net.utility.draw import *

from net.ops.rcnn_loss_op   import *
from net.ops.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels
from net.ops.rcnn_target_op import rcnn_target
from net.ops.rcnn_nms_op    import rcnn_nms
from net.ops.rcnn_nms_op    import draw_rcnn_berfore_nms, draw_rcnn_after_nms_top, draw_rcnn_after_nms_surround

from net.ops.rpn_loss_op   import *
from net.ops.rpn_target_op import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.ops.rpn_target_op import make_bases, make_anchors, rpn_target
from net.ops.rpn_nms_op    import draw_rpn_before_nms, draw_rpn_after_nms

from didi_data.lidar          import *
from didi_data.lidar_surround import *
from didi_data.lidar_top      import *


from didinet import *

#---------------------------------------------------------------------------------------------
#  todo:
#    -- fix anchor index
#    -- 3d box prameterisation
#    -- batch renormalisation
#    -- multiple image training


#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017


def load_dummy_datas():
    is_debug = 0

    data_dir='/root/share/project/didi/data/didi/didi-2/Out/1/15/'
    lidars    =[]
    tops      =[]
    surrounds =[] ## 360 view
    rgbs      =[]
    #radars    =[] ## <todo> to be added later ... maybe can be added as separate channel to lidar data above
    #              ## <todo> since both are some kind of 3d point cloud
    gt_labels  =[]
    gt_boxes3d =[]
    top_images     =[]  ## for visualisation only
    surround_images=[]
    timestamps =[]

    if is_debug:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(500, 500))

    t=0
    for file in sorted(glob.glob(data_dir + '/lidar/*.npy')):
        #if t==1: continue  #load only one for debug
        name = os.path.basename(file).replace('.npy','')

        print ( 'reading at timestamp=%s'%name, end='\r', flush=True)
        lidar = np.load(data_dir+'/lidar/%s.npy'%name)
        top      = np.load(data_dir+'/processed/lidar_top/%s.npy'%name)
        surround = np.load(data_dir+'/processed/lidar_surround/%s.npy'%name)
        rgb   = np.zeros((1,1),dtype=np.float32)  ##<todo> this is dummy. to added after knowning how to do synchronisation
        #radar = np.zeros((1,1),dtype=np.float32)

        gt_box3d = np.load(data_dir+'/processed/gt_boxes3d/%s.npy'%name)
        if len(gt_box3d)==0:
            continue  #<todo> currently ignore pure negative train image

        #assume one box ... exlcude out of range
        if 1:
            min_x, max_x = np.min(gt_box3d[0,:,0]), np.max(gt_box3d[0,:,0])
            min_y, max_y = np.min(gt_box3d[0,:,1]), np.max(gt_box3d[0,:,1])
            if min_x<-20 or max_x>20 or min_y<-20 or max_y>20 : continue

        #<todo> set center to zero .... this is the car with the lidar
        #top[-1:1,-2:2,:]=0

        gt_label = np.ones((len(gt_box3d)), dtype=np.int32) #<todo> assume one class

        top_image      = cv2.imread(data_dir+'/processed/lidar_top_img/%s.png'%name,1)
        surround_image = cv2.imread(data_dir+'/processed/lidar_surround_img/%s.png'%name,1)


        lidars.append(lidar)
        tops.append(top)
        surrounds.append(surround)
        rgbs.append(rgb)
        #radars.append(radar)
        gt_labels.append(gt_label)
        gt_boxes3d.append(gt_box3d)
        top_images.append(top_image)
        surround_images.append(surround_image)
        timestamps.append(name)
        t = t+1

        # explore dataset:
        if is_debug:

            print (gt_box3d)
            #projections = box3d_to_rgb_projections(gt_box3d)
            #rgb1 = draw_rgb_projections(rgb, projections, color=(255,255,255), thickness=2)
            #top_image1 = draw_box3d_on_top(top_image, gt_box3d, color=(255,255,255), thickness=2)

            imshow('surround_image',surround_image)
            imshow('top_image',top_image)

            mlab.clf(fig)
            draw_didi_lidar  (fig, lidar)
            draw_didi_boxes3d(fig, gt_box3d)
            azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
            mlab.view(azimuth,elevation,distance,focalpoint)

            mlab.show(1)
            cv2.waitKey(0)

            pass


    ##exit(0)
    ##mlab.close(all=True)
    return  lidars,  tops,  surrounds, rgbs,  gt_labels, gt_boxes3d, top_images, surround_images, timestamps


def project_to_roi3d(top_rois):
    num = len(top_rois)
    rois3d = np.zeros((num,8,3))
    rois3d = top_box_to_box3d(top_rois[:,1:5])
    return rois3d



def project_to_rgb_roi(rois3d):

    #<todo> : this is dummy function. you should use lidar to camera projection matrix to make camera roi
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)

    return rois



def  project_to_surround_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    rois[:,1:5] = box3d_to_surround_box(rois3d)
    return rois










def run_train():

    # output dir, etc
    out_dir = '/root/share/out/didi/zzz2'

    os.makedirs(out_dir +'/tf', exist_ok=True)
    os.makedirs(out_dir +'/check_points', exist_ok=True)
    log = Logger(out_dir+'/log.txt',mode='a')

    initial_model = '/root/share/out/didi/zzz1/check_points/snap.ckpt-0'  #None for no pretrained model


    #lidar data -----------------
    if 1:
        ratios=np.array([0.5, 1, 2], dtype=np.float32)
        scales=np.array([2],   dtype=np.float32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 4

        lidars,  tops,  surrounds, rgbs,  gt_labels, gt_boxes3d, top_imgs, surround_imgs, timestamps = load_dummy_datas()
        num_frames = len(lidars)

        top_shape      = tops[0].shape
        surround_shape = surrounds[0].shape
        rgb_shape      = rgbs[0].shape
        top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)
        out_shape=(8,3)  #3d box


        #-----------------------
        #check data
        # if 0:
        #     fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
        #     draw_lidar(lidars[0], fig=fig)
        #     draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
        #     mlab.show(1)
        #     cv2.waitKey(1)



    # set anchor boxes
    num_class = 2 #incude background
    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all  #<todo>
    print ('out_shape=%s'%str(out_shape))
    print ('num_frames=%d'%num_frames)


    #load model ####################################################################################################
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_images      = tf.placeholder(shape=[None, *top_shape  ],          dtype=tf.float32, name='top'  )
    surround_images = tf.placeholder(shape=[None, *surround_shape],       dtype=tf.float32, name='surround')
    rgb_images      = tf.placeholder(shape=[None, *rgb_shape  ],          dtype=tf.float32, name='rgb'  )
    top_rois        = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'      ) #<todo> change to int32???
    surround_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='surround_rois' )
    rgb_rois        = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'      )

    top_stride, top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

    surround_stride, surround_features = surround_feature_net(surround_images)
    rgb_stride,      rgb_features      = rgb_feature_net(rgb_images)

    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
			( [top_features,       top_rois,       6,6,1./top_stride     ],
              [surround_features,  surround_rois,  6,6,1./surround_stride],
			  [rgb_features,       rgb_rois,       0,0,1./rgb_stride     ],  #disable by 0,0
			),
            num_class, out_shape) #<todo>  add non max suppression

    #check that stride is correct
    assert (stride==top_stride)

    #loss ########################################################################################################
    top_inds     = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_ind'    )
    top_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_pos_ind')
    top_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_label'  )
    top_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target' )
    top_cls_loss, top_reg_loss = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    fuse_cls_loss, fuse_reg_loss = rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)


    #solver
    l2 = l2_regulariser(decay=0.0005)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #solver_step = solver.minimize(top_cls_loss+top_reg_loss+l2)
    solver_step = solver.minimize(top_cls_loss + 0.1*top_reg_loss + 0.1*fuse_cls_loss+0.01*fuse_reg_loss+l2)



    max_iter = 20000
    iter_debug = 8
    # start training here  #########################################################################################
    log.write('epoch     iter    rate   |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  |  \n')
    log.write('-------------------------------------------------------------------------------------\n')

    num_ratios = len(ratios)
    num_scales = len(scales)
    fig, axs   = plt.subplots(num_scales,num_ratios)

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()
        if initial_model is not None:
            saver.restore(sess, initial_model)

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        for iter in range(max_iter):
            epoch=1.0*iter
            rate=0.05


            ## generate train image -------------
            idx = np.random.choice(num_frames)     #*10   #num_frames)  #0
            batch_top_images      = tops[idx].reshape(1,*top_shape)
            batch_surround_images = surrounds[idx].reshape(1,*surround_shape)
            batch_rgb_images      = rgbs[idx].reshape(1,*rgb_shape)

            batch_gt_labels    = gt_labels[idx]
            batch_gt_boxes3d   = gt_boxes3d[idx]
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)


			## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds,

                learning_rate:   rate,
                IS_TRAIN_PHASE:  True
            }
            batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)

            ## generate train rois  ------------
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets  = \
                rpn_target ( anchors, inside_inds, batch_gt_labels,  batch_gt_top_boxes)

            batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                 rcnn_target(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )

            batch_rois3d	    = project_to_roi3d       (batch_top_rois)
            batch_surround_rois = project_to_surround_roi(batch_rois3d  )
            batch_rgb_rois      = project_to_rgb_roi     (batch_rois3d  )


            ##debug gt generation
            if 1 and iter%iter_debug==0:
                top_image = top_imgs[idx]
                #rgb       = rgbs[idx]

                # rpn
                img_gt     = top_image.copy()  #*0.75  # make dimmer
                img_label  = top_image.copy()
                img_target = top_image.copy()

                draw_rpn_gt(img_gt, batch_gt_top_boxes, batch_gt_labels)
                draw_rpn_labels (img_label, anchors, batch_top_inds, batch_top_labels )
                draw_rpn_targets(img_target, anchors, batch_top_pos_inds, batch_top_targets)
                imshow('img_rpn_gt',img_gt)
                imshow('img_rpn_label',img_label)
                imshow('img_rpn_target',img_target)

                # rcnn
                img_label  = top_image.copy()
                img_target = top_image.copy()

                draw_rcnn_labels (img_label, batch_top_rois, batch_fuse_labels )
                draw_rcnn_targets(img_target, batch_top_rois, batch_fuse_labels, batch_fuse_targets)
                imshow('img_rcnn_label',img_label)
                imshow('img_rcnn_target',img_target)

                cv2.waitKey(1)




            ## run classification and regression loss -----------
            fd2={
				**fd1,

                top_images:      batch_top_images,
                surround_images: batch_surround_images,
                rgb_images:      batch_rgb_images,

				top_rois:        batch_top_rois,
                surround_rois:   batch_surround_rois,
                rgb_rois:        batch_rgb_rois,

                top_inds:        batch_top_inds,
                top_pos_inds:    batch_top_pos_inds,
                top_labels:      batch_top_labels,
                top_targets:     batch_top_targets,

                fuse_labels:     batch_fuse_labels,
                fuse_targets:    batch_fuse_targets,
            }
            #_, batch_top_cls_loss, batch_top_reg_loss = sess.run([solver_step, top_cls_loss, top_reg_loss],fd2)


            _, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss = \
               sess.run([solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],fd2)

            log.write('%3.1f   %d   %0.4f   |   %0.5f   %0.5f   |   %0.5f   %0.5f  \n' %\
				(epoch, iter, rate, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss))



            #print('ok')
            # debug: ------------------------------------

            if iter%iter_debug==0:
                top_image      = top_imgs[idx]
                surround_image = surround_imgs[idx]

                batch_top_probs,  batch_top_deltas  =  sess.run([ top_probs,  top_deltas  ],fd2)
                batch_fuse_probs, batch_fuse_deltas =  sess.run([ fuse_probs, fuse_deltas ],fd2)

                probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.8)


                ## show rpn score maps
                p = batch_top_probs.reshape( *(top_feature_shape[0:2]), 2*num_bases)
                for n in range(num_bases):

                    pn = p[:,:,2*n+1]*255
                    if num_scales==1 or num_ratios==1:
                        axs[n].cla()
                        axs[n].imshow(pn, cmap='gray', vmin=0, vmax=255)
                    else:
                        r=n%num_scales
                        s=n//num_scales
                        axs[r,s].cla()
                        axs[r,s].imshow(pn, cmap='gray', vmin=0, vmax=255)
                plt.pause(0.01)

				## show rpn(top) nms
                img_rpn_before_nms = top_image.copy()
                img_rpn_after_nms  = top_image.copy()
                draw_rpn_before_nms(img_rpn_before_nms, batch_top_probs, batch_top_deltas, anchors, inside_inds)
                draw_rpn_after_nms (img_rpn_after_nms, batch_proposals, batch_proposal_scores)


                imshow('img_rpn_before_nms',img_rpn_before_nms)
                imshow('img_rpn_after_nms', img_rpn_after_nms )
                cv2.waitKey(1)

                ## show rcnn(fuse) nms
                img_rcnn_before_nms     = top_image.copy()
                img_rcnn_after_nms_top  = top_image.copy()
                img_rcnn_after_nms_surround  = surround_image.copy()

                draw_rcnn_berfore_nms   (img_rcnn_before_nms, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d)
                draw_rcnn_after_nms_top (img_rcnn_after_nms_top, boxes3d, probs)
                draw_rcnn_after_nms_surround (img_rcnn_after_nms_surround, boxes3d, probs)

                imshow('img_rcnn_before_nms',img_rcnn_before_nms)
                imshow('img_rcnn_after_nms_top',img_rcnn_after_nms_top)
                imshow('img_rcnn_after_nms_surround',img_rcnn_after_nms_surround)
                cv2.waitKey(1)

            # save: ------------------------------------
            if iter%500==0:
                #saver.save(sess, out_dir + '/check_points/%06d.ckpt'%iter)  #iter
                saver.save(sess, out_dir + '/check_points/snap.ckpt',global_step=0)  #iter







## test final results -------------------------------------------------------------------------------------
def run_test():

    # output dir, etc
    out_dir = '/root/share/out/didi/zzz2'
    os.makedirs(out_dir +'/results/top', exist_ok=True)
    os.makedirs(out_dir +'/results/surround', exist_ok=True)
    os.makedirs(out_dir +'/results/lidar', exist_ok=True)



    os.makedirs(out_dir +'/tf', exist_ok=True)
    os.makedirs(out_dir +'/check_points', exist_ok=True)
    log = Logger(out_dir+'/log.txt',mode='a')
    initial_model = '/root/share/out/didi/zzz2/check_points/snap.ckpt-0'  #None None  #


    #lidar data -----------------
    if 1:
        ratios=np.array([0.5, 1, 2], dtype=np.float32)
        scales=np.array([2],   dtype=np.float32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 4

        lidars,  tops,  surrounds, rgbs,  gt_labels, gt_boxes3d, top_imgs, surround_imgs, timestamps = load_dummy_datas()
        num_frames = len(lidars)

        top_shape      = tops[0].shape
        surround_shape = surrounds[0].shape
        rgb_shape      = rgbs[0].shape
        top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)
        out_shape=(8,3)  #3d box


        #-----------------------
        #check data
        # if 0:
        #     fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
        #     draw_lidar(lidars[0], fig=fig)
        #     draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
        #     mlab.show(1)
        #     cv2.waitKey(1)



    # set anchor boxes
    num_class = 2 #incude background
    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all  #<todo>
    print ('out_shape=%s'%str(out_shape))
    print ('num_frames=%d'%num_frames)


    #load model ####################################################################################################
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_images      = tf.placeholder(shape=[None, *top_shape  ],          dtype=tf.float32, name='top'  )
    surround_images = tf.placeholder(shape=[None, *surround_shape],       dtype=tf.float32, name='surround')
    rgb_images      = tf.placeholder(shape=[None, *rgb_shape  ],          dtype=tf.float32, name='rgb'  )
    top_rois        = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'      ) #<todo> change to int32???
    surround_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='surround_rois' )
    rgb_rois        = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'      )

    top_stride, top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

    surround_stride, surround_features = surround_feature_net(surround_images)
    rgb_stride,      rgb_features      = rgb_feature_net(rgb_images)

    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
			( [top_features,       top_rois,       6,6,1./top_stride     ],
              [surround_features,  surround_rois,  6,6,1./surround_stride],
			  [rgb_features,       rgb_rois,       0,0,1./rgb_stride     ],  #disable by 0,0
			),
            num_class, out_shape) #<todo>  add non max suppression

    #check that stride is correct
    assert (stride==top_stride)




    is_show = 1
    # start testing here  #########################################################################################

    num_ratios = len(ratios)
    num_scales = len(scales)
    fig, axs   = plt.subplots(num_scales,num_ratios)

    mfig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(500, 500))


    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()
        if initial_model is not None:
            saver.restore(sess, initial_model)

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0


        for idx in range(num_frames):
            batch_top_images      = tops[idx].reshape(1,*top_shape)
            batch_surround_images = surrounds[idx].reshape(1,*surround_shape)
            batch_rgb_images      = rgbs[idx].reshape(1,*rgb_shape)

            batch_gt_labels    = gt_labels[idx]
            batch_gt_boxes3d   = gt_boxes3d[idx]
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)


			## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds,
                IS_TRAIN_PHASE:  True
            }
            batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)
            batch_top_rois = batch_proposals

            batch_rois3d	    = project_to_roi3d       (batch_top_rois)
            batch_surround_rois = project_to_surround_roi(batch_rois3d  )
            batch_rgb_rois      = project_to_rgb_roi     (batch_rois3d  )


            ## run classification and regression  -----------
            fd2={
				**fd1,

                top_images:      batch_top_images,
                surround_images: batch_surround_images,
                rgb_images:      batch_rgb_images,

				top_rois:        batch_top_rois,
                surround_rois:   batch_surround_rois,
                rgb_rois:        batch_rgb_rois,

            }
            batch_top_probs,  batch_top_deltas  =  sess.run([ top_probs,  top_deltas  ],fd2)
            batch_fuse_probs, batch_fuse_deltas =  sess.run([ fuse_probs, fuse_deltas ],fd2)

            probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.9)


            #print('ok')
            # debug: ------------------------------------
            if is_show == 1:
                top_image      = top_imgs[idx]
                surround_image = surround_imgs[idx]
                lidar = lidars[idx]

                ## show on lidar
                mlab.clf(mfig)
                draw_didi_lidar(mfig, lidar, is_grid=1, is_axis=1)
                if len(boxes3d)!=0:
                    draw_didi_boxes3d(mfig, boxes3d)
                azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
                mlab.view(azimuth,elevation,distance,focalpoint)
                mlab.show(1)


                ## show rpn score maps
                p = batch_top_probs.reshape( *(top_feature_shape[0:2]), 2*num_bases)
                for n in range(num_bases):

                    pn = p[:,:,2*n+1]*255
                    if num_scales==1 or num_ratios==1:
                        axs[n].cla()
                        axs[n].imshow(pn, cmap='gray', vmin=0, vmax=255)
                    else:
                        r=n%num_scales
                        s=n//num_scales
                        axs[r,s].cla()
                        axs[r,s].imshow(pn, cmap='gray', vmin=0, vmax=255)
                plt.pause(0.01)

				## show rpn(top) nms
                img_rpn_before_nms = top_image.copy()
                img_rpn_after_nms  = top_image.copy()
                draw_rpn_before_nms(img_rpn_before_nms, batch_top_probs, batch_top_deltas, anchors, inside_inds)
                draw_rpn_after_nms (img_rpn_after_nms, batch_proposals, batch_proposal_scores)


                imshow('img_rpn_before_nms',img_rpn_before_nms)
                imshow('img_rpn_after_nms', img_rpn_after_nms )
                cv2.waitKey(1)

                ## show rcnn(fuse) nms
                img_rcnn_before_nms     = top_image.copy()
                img_rcnn_after_nms_top  = top_image.copy()
                img_rcnn_after_nms_surround  = surround_image.copy()

                draw_rcnn_berfore_nms   (img_rcnn_before_nms, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d)
                draw_rcnn_after_nms_top (img_rcnn_after_nms_top, boxes3d, probs)
                draw_rcnn_after_nms_surround (img_rcnn_after_nms_surround, boxes3d, probs)

                imshow('img_rcnn_before_nms',img_rcnn_before_nms)
                imshow('img_rcnn_after_nms_top',img_rcnn_after_nms_top)
                imshow('img_rcnn_after_nms_surround',img_rcnn_after_nms_surround)


                #save
                name=timestamps[idx]
                cv2.imwrite(out_dir +'/results/top/%s.png'%name, img_rcnn_after_nms_top)
                cv2.imwrite(out_dir +'/results/surround/%s.png'%name, img_rcnn_after_nms_surround)
                mlab.savefig(out_dir +'/results/lidar/%s.png'%name,figure=mfig)

                if idx==0: cv2.waitKey(0)

    #make movie
    dir_to_avi(out_dir +'/results/top.avi', out_dir +'/results/top')
    dir_to_avi(out_dir +'/results/surround.avi', out_dir +'/results/surround')
    dir_to_avi(out_dir +'/results/lidar.avi', out_dir +'/results/lidar')


## main function ##########################################################################


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ##run_train()
    run_test()
