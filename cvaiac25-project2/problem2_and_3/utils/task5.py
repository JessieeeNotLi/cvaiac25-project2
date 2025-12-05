import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    
    order = np.argsort(-score, kind='quicksort')
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]
        ious = get_iou(pred[i:i+1], pred[rest])[0]

        keep_mask = ious <= threshold
        order = rest[keep_mask]

    s_f = pred[keep]
    c_f = score[keep][: , None]
    return s_f, c_f