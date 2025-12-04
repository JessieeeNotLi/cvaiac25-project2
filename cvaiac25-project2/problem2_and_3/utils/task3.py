import numpy as np

from .task1 import get_iou

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (bool) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    IoU = get_iou(pred, target)                   
    best_gt_idx = np.argmax(IoU, axis=1)          
    best_iou = IoU[np.arange(pred.shape[0]), best_gt_idx]

    fg_idx = np.where(best_iou >= config['t_fg_lb'])[0]
    bg_idx = np.where(best_iou <  config['t_bg_up'])[0]

    f_n, b_n = len(fg_idx), len(bg_idx)

    f_idx = []
    b_idx = []

    total_samples = 64

    if b_n > 0:
        
        if f_n > 0:
            # foreground and background exist
            fg_samples = min(f_n, config['num_fg_sample'])
            bg_samples = total_samples - fg_samples

            f_choose = np.random.choice(f_n, fg_samples, replace=(f_n < fg_samples))
            f_idx = fg_idx[f_choose]

        else:
            # only background exist
            fg_samples = 0
            bg_samples = total_samples


        bg_iou = best_iou[bg_idx]
        easy_mask = bg_iou < config['t_bg_hard_lb']

        bg_easy_idx = bg_idx[easy_mask]     
        bg_hard_idx = bg_idx[~easy_mask]   

        n_easy, n_hard = len(bg_easy_idx), len(bg_hard_idx)

        if n_easy > 0:

            if n_hard > 0:
                # there is at least one of both
                half_bg = bg_samples // 2
                easy_choose = np.random.choice(n_easy, half_bg, replace=(n_easy < half_bg))

                remaining = bg_samples - half_bg
                hard_choose = np.random.choice(n_hard, remaining, replace=(n_hard < remaining))

                easy_idx = bg_easy_idx[easy_choose]
                hard_idx = bg_hard_idx[hard_choose]
                b_idx = np.concatenate([easy_idx, hard_idx])

            else:
                # no hard samples, sample all from easy
                easy_choose = np.random.choice(n_easy, bg_samples, replace=(n_easy < bg_samples))
                b_idx = bg_easy_idx[easy_choose]
        else:
            # no easy samples, sample all from hard
            hard_choose = np.random.choice(n_hard, bg_samples, replace=(n_hard < bg_samples))
            b_idx = bg_hard_idx[hard_choose]

    else:
        # background does not exist
        f_choose = np.random.choice(f_n, total_samples, replace=(f_n < total_samples))
        f_idx = fg_idx[f_choose]
    
    sampled_indices = np.concatenate([f_idx, b_idx])


    sampled_ious = best_iou[sampled_indices]
    assigned_targets = target[best_gt_idx[sampled_indices]]
    xyz_out  = xyz[sampled_indices]
    feat_out = feat[sampled_indices]

    return assigned_targets, xyz_out, feat_out, sampled_ious