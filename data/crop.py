import numpy as np
from scipy.misc import imread
import os
import errno

CAM = 2

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    #points = points[:, :3]  # exclude luminance
    return points

def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #
    P = np.array(lines[CAM]).reshape(3,4)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pts3d = pts3d_raw
    # Reflectance > 0
    indices = pts3d[:, 3] > 0
    pts3d = pts3d[indices ,:]
    pts3d[:,3] = 1
    return pts3d.transpose(), indices

def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2,:]>=0)
    pts2d_cam = Prect.dot(pts3d_cam[:,idx])
    return pts3d[:, idx], pts2d_cam/pts2d_cam[2,:], idx


def align_img_and_pc(img_dir, pc_dir, calib_dir):
    
    img = imread(img_dir)
    pts = load_velodyne_points( pc_dir )
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)

    pts3d, indices = prepare_velo_points(pts)
    pts3d_ori = pts3d.copy()
    reflectances = pts[indices, 3]
    pts3d, pts2d_normed, idx = project_velo_points_in_img( pts3d, Tr_velo_to_cam, R_cam_to_rect, P  )
    #print reflectances.shape, idx.shape
    reflectances = reflectances[idx]
    #print reflectances.shape, pts3d.shape, pts2d_normed.shape
    assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

    rows, cols = img.shape[:2]

    points = []
    for i in range(pts2d_normed.shape[1]):
        c = int(np.round(pts2d_normed[0,i]))
        r = int(np.round(pts2d_normed[1,i]))
        if c < cols and r < rows and r > 0 and c > 0:
            color = img[r, c, :]
            point = [ pts3d[0,i], pts3d[1,i], pts3d[2,i], reflectances[i], color[0], color[1], color[2], pts2d_normed[0,i], pts2d_normed[1,i]  ]
            points.append(point)

    points = np.array(points)
    return points

if __name__ == '__main__':
    '''
    create folder in following structure
    └── DATA_DIR
           ├── training   <-- training data
           |   ├── image_2
           |   ├── label_2
           |   ├── velodyne
           |   └── calib
           └── validation  <--- evaluation data
           |   ├── image_2
           |   ├── label_2
           |   ├── velodyne
           |   └── calib                 
    '''
    cwd = os.getcwd()

    for path in [os.path.join(cwd,'training'),
                 os.path.join(cwd,'validation'),
                 os.path.join(cwd,'training','image_2'),
                 os.path.join(cwd,'training','label_2'),
                 os.path.join(cwd,'training','velodyne'),
                 os.path.join(cwd,'training','calib'),
                 os.path.join(cwd,'validation','image_2'),
                 os.path.join(cwd,'validation','label_2'),
                 os.path.join(cwd,'validation','velodyne'),
                 os.path.join(cwd,'validation','calib')]:
        try:
            os.mkdir(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    #target root for training and validation 
    training_image_2 = os.path.join(cwd,'training','image_2')
    training_label_2 = os.path.join(cwd,'training','label_2')
    training_velodyne = os.path.join(cwd,'training','velodyne')
    training_calib = os.path.join(cwd,'training','calib')
    validation_image_2 = os.path.join(cwd,'validation','image_2')
    validation_label_2 = os.path.join(cwd,'validation','label_2')
    validation_velodyne = os.path.join(cwd,'validation','velodyne')
    validation_calib = os.path.join(cwd,'validation','calib')

    #-------------------------------------------------------------------------------------------------------------

    #update the root of the data_object_image_2, data_object_velodyne, data_object_calib, label_2(training) and split file(ImageSets)
    data_object_image_dir = './../../'
    data_object_velodyne_dir = './../../'
    data_object_calib_dir = './../../'
    data_object_label_dir = './../../'
    split_dir = './../../'

    #-----------------------------------------------------------------------------------------------------------
    # update the following directories
    IMG_ROOT = os.path.join(data_object_image_dir,'data_object_image_2','training','image_2')
    PC_ROOT = os.path.join(data_object_velodyne_dir,'data_object_velodyne','training','velodyne')
    CALIB_ROOT = os.path.join(data_object_calib_dir,'data_object_calib','training','calib')
    LABEL_ROOT = os.path.join(data_object_label_dir,'training','label_2')
    SPLIT_ROOT = os.path.join(split_dir,'ImageSets')

    # for training set
    for line in open(os.path.join(SPLIT_ROOT,'train.txt'), 'r').readlines():

        #there is a space in every line except the last line of the orginial split file,we need to remove the space
        if(len(line)!=6):
            line = line[:-1]
        img_dir = os.path.join(IMG_ROOT,line+'.png')
        pc_dir = os.path.join(PC_ROOT,line+'.bin')
        calib_dir = os.path.join(CALIB_ROOT,line+'.txt')
        label_dir = os.path.join(LABEL_ROOT,line+'.txt')

        if(os.path.exists(pc_dir) and os.path.exists(img_dir) and os.path.exists(calib_dir)):
            points = align_img_and_pc(img_dir, pc_dir, calib_dir)

            #create output file in ./training/velodyne dir
            output_name = os.path.join(training_velodyne,line+'.bin')
            points[:,:4].astype('float32').tofile(output_name)

        #move image_2 file,label_2 and calib file into training set 
        if(os.path.exists(img_dir)):
            os.rename(img_dir,os.path.join(training_image_2,line+'.png'))
        if(os.path.exists(label_dir)):
            os.rename(label_dir,os.path.join(training_label_2,line+'.txt')) 
        if(os.path.exists(calib_dir)):
            os.rename(calib_dir,os.path.join(training_calib,line+'.txt'))

    # for validation set
    for line in open(os.path.join(SPLIT_ROOT,'val.txt'), 'r').readlines():

        #there is a space in every line except the last line of the orginial split file,we need to remove the space
        if(len(line)!=6):
            line = line[:-1]
        img_dir = os.path.join(IMG_ROOT,line+'.png')
        pc_dir = os.path.join(PC_ROOT,line+'.bin')
        calib_dir = os.path.join(CALIB_ROOT,line+'.txt')
        label_dir = os.path.join(LABEL_ROOT,line+'.txt')

        if(os.path.exists(pc_dir) and os.path.exists(img_dir) and os.path.exists(calib_dir)):
            points = align_img_and_pc(img_dir, pc_dir, calib_dir)

            #create output file in ./validation/velodyne dir
            output_name = os.path.join(validation_velodyne,line+'.bin')
            points[:,:4].astype('float32').tofile(output_name)

        #move image_2 file, label_2 and calib file into validation set 
        if(os.path.exists(img_dir)):
            os.rename(img_dir,os.path.join(validation_image_2,line+'.png'))
        if(os.path.exists(label_dir)):
            os.rename(label_dir,os.path.join(validation_label_2,line+'.txt')) 
        if(os.path.exists(calib_dir)):
            os.rename(calib_dir,os.path.join(validation_calib,line+'.txt'))

'''
for frame in range(0, 7481):
    img_dir = IMG_ROOT + '%06d.png' % frame
    pc_dir = PC_ROOT + '%06d.bin' % frame
    calib_dir = CALIB_ROOT + '%06d.txt' % frame
    if(os.path.exists(pc_dir)):
        points = align_img_and_pc(img_dir, pc_dir, calib_dir)
    
        output_name = PC_ROOT + str(frame) + '.bin'
        points[:,:4].astype('float32').tofile(output_name)
'''






