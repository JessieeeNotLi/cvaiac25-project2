import numpy as np
from shapely.geometry import Polygon

def label2corners(label):
    '''
    Task 1
    input:
        label (N,7) 3D bounding boxes (x, y, z, h, w, l, ry)
    output:
        corners (N,8,3) corner coordinates in rectified reference frame
    '''

    corners = np.zeros((label.shape[0], 8, 3))  # output array

    for i, (x, y, z, h, w, l, ry) in enumerate(label):

        #KITTI -y is up
        # 1. Define the bounding box in the local coordinate system
        half_l, half_w = l / 2, w / 2
        box = np.array([
            [ half_l,  0,  half_w],
            [ half_l,  0, -half_w],
            [-half_l,  0, -half_w],
            [-half_l,  0,  half_w],
            [ half_l, -h,  half_w],
            [ half_l, -h, -half_w],
            [-half_l, -h, -half_w],
            [-half_l, -h,  half_w],
        ])

        # 2. Rotation around the y-axis (heading ry)
        c, s = np.cos(ry), np.sin(ry)

        R_T = np.array([
            [ c, 0, -s],
            [ 0, 1, 0],
            [s, 0, c]
        ])

        box = box @ R_T

        # 3. Translation to world coordinates
        box += np.array([x, y, z])
        corners[i] = box

    return corners

def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    N, M = pred.shape[0], target.shape[0]
    p_corners, t_corners = label2corners(pred), label2corners(target)

    # Extract bottom face corners and project to BEV      
    p_bev    = p_corners[:, :4, [0, 2]]
    t_bev = t_corners[:, :4, [0, 2]]

    IoU = np.zeros((N, M))

    for i in range(N):
        _, p_y, _, p_h, p_w, p_l, _ = pred[i]

        for j in range(M):
            _, t_y, _, t_h, t_w, t_l, _ = target[j]

            height = min(p_y, t_y) - max(p_y - p_h, t_y - t_h)
            if height <= 0:
                continue
            
            poly_p, poly_t = Polygon(p_bev[i]), Polygon(t_bev[j])
            inter_poly = poly_p.intersection(poly_t)
            base = inter_poly.area
            if base <= 0:
                continue

            intersection = base * height
            union = p_l * p_w * p_h + t_l * t_w * t_h - intersection

            IoU[i, j] = intersection / union

    return IoU

def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''
    iou = get_iou(pred, target)
    TP = np.sum(np.max(iou, axis=0) >= threshold)
    FN = target.shape[0] - TP

    return TP / (TP + FN) if (TP + FN) > 0 else 0.0