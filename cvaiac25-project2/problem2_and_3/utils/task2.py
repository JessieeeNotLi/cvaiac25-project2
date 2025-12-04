import numpy as np

def is_inside(p, pred):
      
    cx, cy, cz, h, w, l, ry = box

    # 1. Translate point into box coordinate frame
    px, py, pz = p - np.array([cx, cy, cz])

    c = np.cos(ry)
    s = np.sin(ry)

    x_local =  c * px + -s * pz
    y_local =  py
    z_local =  s * px +  c * pz

    return (-h <= y_local <= 0) and (-l/2 <= x_local <= l/2) and (-w/2 <= z_local <= w/2)

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

    for box in pred_enlarged:
        idx = []

        for i in range(xyz.shape[0]):
            if is_inside(xyz[i], box):
                idx.append(i)
        
        n = len(idx)
        if n > 0:
            idxs = np.random.choice(n, M, replace=(n < M))
            idx = np.array(idx)[idxs]
            
            pooled_xyz.append(xyz[idx])
            pooled_feat.append(feat[idx])
            valid_pred.append(box)

    return valid_pred, pooled_xyz, pooled_feat