import numpy as np

global_orient[:,:] = global_orient[0,:]
transl[:,:] = transl[0,:]

if (
    speaker == "oliver" or
    speaker == "seth" or
    speaker == "chemistry"
):
    pose_type='sitting'
else:
    pose_type='standing'


if pose_type == 'standing':
    ref_pose = np.zeros(55 * 3)
elif pose_type == 'sitting':
    ref_pose = np.array([
        0.0, 0.0, 0.0, -1.1826512813568115, 0.23866955935955048, 0.15146760642528534, -1.2604516744613647,
        -0.3160211145877838, -0.1603458970785141, 0.0, 0.0, 0.0, 1.1654603481292725, 0.0, 0.0,
        1.2521806955337524, 0.041598282754421234, -0.06312154978513718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0])

body_pose = body_pose_axis.reshape(bs, 63)
for i in [1, 2, 4, 5, 7, 8, 10, 11]:
    body_pose[:, (i - 1) * 3 + 0] = ref_pose[(i) * 3 + 0]
    body_pose[:, (i - 1) * 3 + 1] = ref_pose[(i) * 3 + 1]
    body_pose[:, (i - 1) * 3 + 2] = ref_pose[(i) * 3 + 2]