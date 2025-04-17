import numpy as np
from deepfusion.matching import linear_assignment
from deepfusion.cost_function import iou_2d

def data_fusion(detections_3D_camera, detection_2D, detection_3Dto2D, additional_info):
    """
    :param detections_3D_camera:  (N,7) - 3D bounding box in camera coords: [x,y,z,rot_y,l,w,h] 
    :param detection_2D:          (M,4) - 2D bounding boxes from camera in [x1, y1, x2, y2]
    :param detection_3Dto2D:      (N,4) - 2D bounding boxes projected from 3D detection
    :param additional_info:       (N,7) - e.g. orientation, detection scores, object type, etc.
    :return:
        detection_3D_fusion: { 'dets_3d_fusion': [...], 'dets_3d_fusion_info': [...] }
        detection_3D_only:   { 'dets_3d_only': [...], 'dets_3d_only_info': [...] }
        detection_2D_only:   [ [...], [...], ... ]
    """
    iou_threshold = 0.3
    if len(detection_2D) == 0 or len(detection_3Dto2D) == 0:
        # If no 2D or no 3Dto2D, then either everything is unmatched or...
        detection_3D_fusion = {'dets_3d_fusion': [], 'dets_3d_fusion_info': []}
        detection_3D_only = {
            'dets_3d_only': detections_3D_camera.tolist(),
            'dets_3d_only_info': additional_info.tolist()
        } if len(detection_3Dto2D) > 0 else {'dets_3d_only': [], 'dets_3d_only_info': []}
        detection_2D_only = detection_2D.tolist() if len(detection_2D) > 0 else []
        return detection_3D_fusion, detection_3D_only, detection_2D_only

    # Construct IoU matrix
    iou_matrix = np.zeros((len(detection_2D), len(detection_3Dto2D)), dtype=np.float32)
    for i, det2d in enumerate(detection_2D):
        for j, det3d2d in enumerate(detection_3Dto2D):
            iou_matrix[i, j] = iou_2d(det2d, det3d2d)

    # Hungarian / linear assignment
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))

    matched = []
    unmatched_2d = []
    unmatched_3dto2d = []

    for d in range(len(detection_2D)):
        if d not in matched_indices[:, 0]:
            unmatched_2d.append(d)
    for t in range(len(detection_3Dto2D)):
        if t not in matched_indices[:, 1]:
            unmatched_3dto2d.append(t)
    # filter out any low iou matches
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_2d.append(m[0])
            unmatched_3dto2d.append(m[1])
        else:
            matched.append(m.reshape(1,2))
    if len(matched) == 0:
        matched = np.empty((0,2), dtype=int)
    else:
        matched = np.concatenate(matched, axis=0)

    # Prepare final outputs
    detection_2D_fusion = []
    detection_3Dto2D_fusion = []
    detection_3D_fusion_vals = []
    detection_3D_fusion_info = []

    for (d_2d_idx, d_3d_idx) in matched:
        detection_2D_fusion.append(detection_2D[d_2d_idx].tolist())
        detection_3Dto2D_fusion.append(detection_3Dto2D[d_3d_idx].tolist())
        detection_3D_fusion_vals.append(detections_3D_camera[d_3d_idx].tolist())
        detection_3D_fusion_info.append(additional_info[d_3d_idx].tolist())

    detection_3D_fusion = {
        'dets_3d_fusion': detection_3D_fusion_vals,
        'dets_3d_fusion_info': detection_3D_fusion_info
    }

    detection_2D_only = [detection_2D[i].tolist() for i in unmatched_2d]
    detection_3D_only_vals = []
    detection_3D_only_info = []
    for idx in unmatched_3dto2d:
        detection_3D_only_vals.append(detections_3D_camera[idx].tolist())
        detection_3D_only_info.append(additional_info[idx].tolist())
    detection_3D_only = {
        'dets_3d_only': detection_3D_only_vals,
        'dets_3d_only_info': detection_3D_only_info
    }

    return detection_3D_fusion, detection_3D_only, detection_2D_only

