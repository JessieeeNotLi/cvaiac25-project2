import numpy as np

def is_inside(p, pred):
      
    cx, cy, cz, h, w, l, ry = pred

    # 1. Translate point into box coordinate frame
    px, py, pz = p - np.array([cx, cy, cz])

    c = np.cos(ry)
    s = np.sin(ry)

    x_local =  c * px + -s * pz
    y_local =  py
    z_local =  s * px +  c * pz

    return (-h <= y_local <= 0) and (-l/2 <= x_local <= l/2) and (-w/2 <= z_local <= w/2)

def points_in_box(xyz, box): #vectorized version of inside with numpy
    cx,cy,cz,h,w,l,ry = box

    pts = xyz - np.array([cx,cy,cz])
    px,py,pz = pts[:,0], pts[:,1], pts[:,2]

    c = np.cos(ry)
    s = np.sin(ry)

    x_local =  c * px + -s * pz
    y_local =  py
    z_local =  s * px +  c * pz

    inside = (
        (-h <= y_local) & (y_local <= 0) &
        (-l / 2 <= x_local) & (x_local <= l / 2) &
        (-w / 2 <= z_local) & (z_local <= w / 2)
    )

    return np.where(inside)[0]
def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly M=512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    pred_enlarged = np.copy(pred)
    delta = config['delta']
    M = config['max_points']

    # enlarge boxes
    pred_enlarged[:,3:6] += 2 * delta

    valid_pred = []
    pooled_xyz = []
    pooled_feat = []

    for box_enlarged, box in zip(pred_enlarged, pred):

        idxs= points_in_box(xyz, box_enlarged)  #indices of points inside the box
        n = len(idxs)
        if n == 0:
            continue
        
        if n >= M:
            chosen_idxs = np.random.choice(idxs, M, replace=False)
        else:
            all_idxs = np.random.choice(idxs, n, replace=False)
            extra_idxs = np.random.choice(idxs, M - n, replace=True)
            chosen_idxs = np.concatenate([all_idxs, extra_idxs], axis=0)

        pooled_xyz.append(xyz[chosen_idxs])
        pooled_feat.append(feat[chosen_idxs])
        valid_pred.append(box)

    if len(valid_pred) == 0:
        return np.zeros((0,7)), np.zeros((0,M,3)), np.zeros((0,M,feat.shape[1]))
    
    valid_pred = np.asarray(valid_pred, dtype=pred.dtype)  
    pooled_xyz = np.stack(pooled_xyz, axis=0)             
    pooled_feat = np.stack(pooled_feat, axis=0)            
    return valid_pred, pooled_xyz, pooled_feat