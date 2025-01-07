import numpy as np
import os

from deepfusionmot.tracker import Tracker
from deepfusionmot.detection import Detection_3D_Fusion, Detection_3D_only, Detection_2D
from deepfusionmot.kitti_oxts import load_oxts  # if you need ego-motion
# ... any other imports

class DeepFusionMOT:
    def __init__(self, cfg, category):
        self.tracker = Tracker(cfg, category)
        self.frame_count = 0
        # If KITTI reorder is relevant to your dataset:
        self.reorder = [3,4,5,6,2,1,0]
        self.reorder_back = [6,5,4,0,1,2,3]
        self.min_frames = cfg[category].min_frames

    def update(self, dets_3d_fusion, dets_2d_only, dets_3d_only, cfg, frame, seq_id):
        """
        The same logic that was in your original update() function.
        """
        dets_3d_fusion_camera = np.array(dets_3d_fusion['dets_3d_fusion'])
        dets_3d_fusion_info = np.array(dets_3d_fusion['dets_3d_fusion_info'])
        dets_3d_only_camera = np.array(dets_3d_only['dets_3d_only'])
        dets_3d_only_info = np.array(dets_3d_only['dets_3d_only_info'])

        # Reorder if needed
        if dets_3d_fusion_camera.shape[0] > 0:
            dets_3d_fusion_camera = dets_3d_fusion_camera[:, self.reorder]
        if dets_3d_only_camera.shape[0] > 0:
            dets_3d_only_camera = dets_3d_only_camera[:, self.reorder]

        # Convert to detection objects
        dets_3d_fusion_camera = [
            Detection_3D_Fusion(dets_3d_fusion_camera[i], dets_3d_fusion_info[i])
            for i in range(len(dets_3d_fusion_camera))
        ]
        dets_3d_only_camera = [
            Detection_3D_only(dets_3d_only_camera[i], dets_3d_only_info[i])
            for i in range(len(dets_3d_only_camera))
        ]
        dets_2d_only = [Detection_2D(i) for i in dets_2d_only]

        # Predict
        self.tracker.predict_2d()
        self.tracker.predict_3d()

        # (Optional) Ego-motion if needed
        # ...

        # Update
        self.tracker.update(dets_3d_fusion_camera, dets_3d_only_camera, dets_2d_only)

        self.frame_count += 1
        outputs = []
        for track in self.tracker.tracks_3d:
            if track.is_confirmed() or self.frame_count <= self.min_frames:
                bbox = np.array(track.pose[self.reorder_back])
                # track.pose is [x,y,z,rot,l,w,h], reorder_back -> [h,w,l,x,y,z,rot]
                # plus any additional info
                out = np.concatenate(([track.track_id_3d], bbox, track.additional_info), axis=0)
                outputs.append(out.reshape(1, -1))

        if len(outputs) > 0:
            outputs = np.vstack(outputs)
        else:
            outputs = np.empty((0,9))
        return outputs

