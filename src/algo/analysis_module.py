#TODO: comparing sliding window and not just one slice
#TODO: Make a class!

import numpy as np

pose_idxs = {
    'BODY': [11, 12, 23, 24],
    'LEFT_HAND': [13, 15, 17, 19, 21],
    'RIGHT_HAND': [14, 16, 18, 20, 22],
    'LEFT_FOOT': [25, 27, 29, 31],
    'RIGHT_FOOT': [26, 28, 30, 32],
    'HEAD': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # not in use
}

#POSE_CHECK = ['LEFT_HAND', 'RIGHT_HAND', 'LEFT_FOOT', 'RIGHT_FOOT']
POSE_CHECK = ['LEFT_HAND', 'RIGHT_HAND']


def get_bias_var(pose, pts=pose_idxs['BODY']):
    x = [pose.landmark[pt].x for pt in pts]
    y = [pose.landmark[pt].y for pt in pts]
    z = [pose.landmark[pt].z for pt in pts]
    bias = np.array([np.mean(x), np.mean(y), np.mean(z)])
    var = np.array([np.var(x), np.var(y), np.var(z)])
    return bias, var

def calc_norm_landmarks(pt1, pt2):
    if pt1.visibility > 0.3 and pt2.visibility > 0.3:
        dx = pt1.x - pt2.x
        dy = pt1.y - pt2.y
        dz = pt1.z - pt2.z
        return np.linalg.norm(np.array([dx, dy, dz]))
    else:
        return np.nan

def calc_distance(pose1, pose2):
    avg_dist, vec_dist = None, []
    for body_part in POSE_CHECK:
        dist = []
        for pt in pose_idxs[body_part]:
            dist.append(calc_norm_landmarks(pose1.landmark[pt], pose2.landmark[pt]))
        vec_dist.append(np.nanmean(np.array(dist)))
    avg_dist = np.nanmean(np.array(vec_dist))
    return avg_dist, vec_dist

def normalize_landmark(landmark, bias1, bias2, var):
    landmark.x = ((landmark.x - bias1[0]) / np.sqrt(var[0])) + bias2[0]
    landmark.y = ((landmark.y - bias1[1]) / np.sqrt(var[1])) + bias2[1]
    landmark.z = ((landmark.z - bias1[2]) / np.sqrt(var[2])) + bias2[2]
    return landmark

def normalize_poses(pose1, pose2):
    bias1, var1 = get_bias_var(pose1, pose_idxs['BODY'])
    bias2, var2 = get_bias_var(pose2, pose_idxs['BODY'])
    var = var1 / var2
    for landmark in pose1.landmark:
        landmark = normalize_landmark(landmark, bias1, bias2, var)


    return pose1, pose2

def compare_pose(pose1, pose2):
    avg_dist, vec_dist, pose1_2d_norm, pose2_2d_norm = None, [], None, None
    if pose1 and pose2 and pose1[0] and pose2[0]:
        pose1_2d, pose1_3d = pose1[0], pose1[1]
        pose2_2d, pose2_3d = pose2[0], pose2[1]
        if pose1_2d and pose2_2d:
            pose1_3d_norm, pose2_3d_norm = normalize_poses(pose1_3d, pose2_3d)
            pose1_2d_norm, pose2_2d_norm = normalize_poses(pose1_2d, pose2_2d)
            avg_dist, vec_dist = calc_distance(pose1_3d_norm, pose2_3d_norm)  # left hand, right hand, left foot, right foot
    return avg_dist, vec_dist, pose1_2d_norm, pose2_2d_norm


