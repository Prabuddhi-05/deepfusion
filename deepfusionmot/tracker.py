import numpy as np
from deepfusionmot.track_3d import Track_3D
from deepfusionmot.track_2d import Track_2D
from deepfusionmot.kalman_filter_2d import KalmanFilter
from deepfusionmot.kalman_filter_3d import KalmanBoxTracker
from deepfusionmot.matching import associate_dets_to_trks_fusion, associate_2D_to_3D_tracking

class Tracker:
    def __init__(self, cfg, category):
        self.cfg = cfg
        self.cost_3d = cfg[category].metric_3d
        self.cost_2d = cfg[category].metric_2d
        self.threshold_3d = cfg[category]["cost_function"][self.cost_3d]
        self.threshold_2d = cfg[category]["cost_function"][self.cost_2d]
        self.max_age = cfg[category].max_ages
        self.min_frames = cfg[category].min_frames
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0
        self.track_id_2d = 1
        self.unmatch_tracks_3d = []
        self.kf_2d = KalmanFilter()

    def predict_3d(self):
        for t in self.tracks_3d:
            t.predict_3d(t.kf_3d)

    def predict_2d(self):
        for t in self.tracks_2d:
            t.predict_2d(self.kf_2d)

    #def ego_motion_compensation(self, ...):
        #pass  # If needed

    def update(self, dets_3d_fusion, dets_3d_only, dets_2d_only):
        """
        4-level data association logic (1st: 3D fusion, 2nd: 3D-only, 3rd: 2D-only, 4th: unmatched 2D w/ unmatched 3D).
        """
        # 1) Associate 3D fusion with tracks
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
            dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d'
        )
        for det_i, trk_i in matched_fusion_idx:
            self.tracks_3d[trk_i].update_3d(dets_3d_fusion[det_i])
        # Mark missed
        for trk_i in unmatched_trks_fusion_idx:
            self.tracks_3d[trk_i].mark_missed()
        # Create new
        for det_i in unmatched_dets_fusion_idx:
            self.initiate_trajectory_3d(dets_3d_fusion[det_i])

        # 2) 3D-only
        unmatched_tracks_level1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        matched_only_idx, unmatched_dets_only_idx, _ = associate_dets_to_trks_fusion(
            dets_3d_only, unmatched_tracks_level1, self.cost_3d, self.threshold_3d, metric='match_3d'
        )
        for det_i, trk_i in matched_only_idx:
            # find the actual track in self.tracks_3d
            track_obj = unmatched_tracks_level1[trk_i]
            track_obj.update_3d(dets_3d_only[det_i])
        for det_i in unmatched_dets_only_idx:
            self.initiate_trajectory_3d(dets_3d_only[det_i])

        # 3) 2D-only
        matched, unmatch_trks, unmatch_dets = associate_dets_to_trks_fusion(
            self.tracks_2d, dets_2d_only, self.cost_2d, self.threshold_2d, metric='match_2d'
        )
        for t_i, d_i in matched:
            self.tracks_2d[t_i].update_2d(self.kf_2d, dets_2d_only[d_i])
        for t_i in unmatch_trks:
            self.tracks_2d[t_i].mark_missed()
        for d_i in unmatch_dets:
            self.initiate_trajectory_2d(dets_2d_only[d_i])
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        # 4) unmatched 2D with unmatched 3D
        unmatched_3d = [t for t in self.tracks_3d if t.is_unmatched()]  # e.g. time_since_update>0
        matched_track_2d, _ = associate_2D_to_3D_tracking(self.tracks_2d, unmatched_3d, self.threshold_2d)
        for t2d_i, t3d_i in matched_track_2d:
            # Merge track states
            pass

        # Clean up
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]

    def initiate_trajectory_3d(self, detection):
        kf_3d = KalmanBoxTracker(detection.bbox)
        new_track = Track_3D(detection.bbox, kf_3d, self.track_id_3d, self.min_frames, self.max_age, detection.additional_info)
        #new_track = Track_3D(detection.bbox, kf_3d, self.track_id_3d, self.min_frames, self.max_age, detection.additional_info)
        self.tracks_3d.append(new_track)
        self.track_id_3d += 2

    def initiate_trajectory_2d(self, detection):
        mean, cov = self.kf_2d.initiate(detection.to_xyah())
        new_track = Track_2D(mean, cov, self.track_id_2d, self.min_frames, self.max_age)
        self.tracks_2d.append(new_track)
        self.track_id_2d += 2

