import copy
import numpy as np
from deepfusionmot.calibration import Calibration

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


class TransformationKitti(Calibration):
    """
    Extended so it can parse either:
      1) a file path (default),
      2) a multi-line calibration string in memory (is_string=True).

    Once parsed, we store:
       self.P2              (3x4)
       self.R0_rect         (3x3)
       self.Tr_lidar_to_cam (4x4), aka Tr_velo_cam
       self.Tr_imu_to_lidar (4x4) if present
    """

    def __init__(self, calib_data, is_string=False):
        if not is_string:
            # Old behavior: treat calib_data as a file path
            super().__init__(calib_data)
        else:
            # New behavior: parse calib_data as multi-line string
            # and set up the calibration matrices in memory
            self.parse_calib_str(calib_data)

    def parse_calib_str(self, calib_str):
        """
        Parse the multi-line calibration text. 
        We'll look for lines with 'P2:', 'R_rect', 'Tr_velo_cam', etc.
        and fill in self.P2, self.R0_rect, self.Tr_lidar_to_cam, etc.
        """
        # Initialize defaults (in case some lines are missing)
        self.P2 = np.eye(3, 4)
        self.R0_rect = np.eye(3)
        self.Tr_lidar_to_cam = np.eye(4)
        self.Tr_imu_to_lidar = np.eye(4)
        # If you have c_u, c_v, f_u, f_v, etc. from parent, define or parse them if needed.

        lines = calib_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('P2:'):
                # e.g. "P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 ..."
                nums = line.replace('P2:', '').split()
                nums = [float(x) for x in nums]
                # P2 is 3x4
                self.P2 = np.array(nums).reshape(3, 4)

            elif line.startswith('R_rect'):
                # e.g. "R_rect 9.999239e-01 9.837760e-03 -7.445048e-03 ..."
                nums = line.replace('R_rect', '').split()
                nums = [float(x) for x in nums]
                # R0_rect is 3x3
                self.R0_rect = np.array(nums).reshape(3, 3)

            elif line.startswith('Tr_velo_cam'):
                # e.g. "Tr_velo_cam 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 ..."
                nums = line.replace('Tr_velo_cam', '').split()
                nums = [float(x) for x in nums]
                # Typically 3x4 in KITTI => embed in 4x4
                M = np.eye(4)
                M[0:3, :] = np.array(nums).reshape(3, 4)
                self.Tr_lidar_to_cam = M

            elif line.startswith('Tr_imu_velo'):
                # e.g. "Tr_imu_velo 9.999976e-01 7.553071e-04 ..."
                nums = line.replace('Tr_imu_velo', '').split()
                nums = [float(x) for x in nums]
                M = np.eye(4)
                M[0:3, :] = np.array(nums).reshape(3, 4)
                self.Tr_imu_to_lidar = np.linalg.inv(M)

            # If you have other lines: "P0:, P1:, R1_rect", etc. parse if needed.

    @property
    def Tr_velo_to_cam_4x4(self):
        """
        Return the 4x4 transform from velodyne->camera.
        Some code calls this "Tr_lidar_to_cam".
        """
        return self.Tr_lidar_to_cam

    # ---------------------------
    #  3d -> 3d transformations
    # ---------------------------
    def project_lidar_to_ref(self, pts_3d_lidar):
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)
        return np.dot(pts_3d_lidar, np.transpose(self.Tr_lidar_to_cam))

    def project_imu_to_lidar(self, pts_3d_imu):
        pts_3d_imu = self.cart2hom(pts_3d_imu)
        return np.dot(pts_3d_imu, np.transpose(self.Tr_imu_to_lidar))

    def project_lidar_to_imu(self, pts_3d_lidar):
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)
        return np.dot(pts_3d_lidar, np.transpose(self.Tr_imu_to_lidar))

    def project_ref_to_lidar(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)
        Tr_cam_to_lidar = np.linalg.inv(self.Tr_lidar_to_cam)
        return np.dot(pts_3d_ref, np.transpose(Tr_cam_to_lidar))

    def project_rect_to_ref(self, pts_3d_rect):
        return np.transpose(np.dot(np.linalg.inv(self.R0_rect), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        return np.transpose(np.dot(self.R0_rect, np.transpose(pts_3d_ref)))

    def project_rect_to_lidaro(self, pts_3d_rect):
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_lidar(pts_3d_ref)

    def project_lidar_to_rect(self, pts_3d_lidar):
        pts_3d_ref = self.project_lidar_to_ref(pts_3d_lidar)
        return self.project_ref_to_rect(pts_3d_ref)

    # ---------------------------
    #  3d -> 2d transformations
    # ---------------------------
    def project_rect_to_image(self, pts_3d_rect):
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_3d_to_image(self, pts_3d_rect):
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_lidar_to_image(self, pts_3d_lidar):
        pts_3d_rect = self.project_lidar_to_rect(pts_3d_lidar)
        return self.project_rect_to_image(pts_3d_rect)

    # ---------------------------
    #  2d -> 3d transformations
    # ---------------------------
    def project_image_to_rect(self, uv_depth):
        # Example usage if your parent class sets c_u, c_v, f_u, f_v
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_lidar(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_lidar(pts_3d_rect)


##################################################
# 3D box functions 
##################################################
def convert_3dbox_to_8corner(bbox3d_input):
    ''' 
    Takes an object's 3D box with the representation of [h,w,l, x,y,z,theta] and
    converts it to the 8 corners of the 3D box in rect camera coords.

    Returns: (8,3) array in rect camera coordinates
    '''
    bbox3d = copy.copy(bbox3d_input)
    R = roty(bbox3d[6])

    h = bbox3d[0]
    w = bbox3d[1]
    l = bbox3d[2]
    cx= bbox3d[3]
    cy= bbox3d[4]
    cz= bbox3d[5]

    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [   0.,   0.,   0.,   0.,  -h,   -h,   -h,   -h]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += cx
    corners_3d[1,:] += cy
    corners_3d[2,:] += cz

    return corners_3d.T  # shape (8,3)


def compute_box_3dto2d(bbox3d_input, calib_data_or_file, from_string=False):
    '''
    Takes [h, w, l, x, y, z, theta] + calibration data (file or string)
    and returns corners_2d: (8,2) array in image coords (or None if behind camera).
    '''
    bbox3d = copy.copy(bbox3d_input)
    R = roty(bbox3d[6])
    l = bbox3d[2]
    w = bbox3d[1]
    h = bbox3d[0]

    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]
    z_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += bbox3d[3]  # x
    corners_3d[1, :] += bbox3d[4]  # y
    corners_3d[2, :] += bbox3d[5]  # z

    # if any corner has z < 0.1 => behind camera
    if np.any(corners_3d[2, :] < 0.1):
        return None

    corners_3d = corners_3d.T  # shape (8,3)

    calib = TransformationKitti(calib_data_or_file, is_string=from_string)

    corners_2d = calib.project_3d_to_image(corners_3d)  # shape (8,2)
    return corners_2d


##################################################
# Additional utility functions
##################################################
def convert_x1y1x2y2_to_xywh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    return np.array([x, y, w, h]).tolist()


def convert_x1y1x2y2_to_tlwh(bbox):
    '''
    :param bbox: x1 y1 x2 y2
    :return: tlwh: top_left x   top_left y    width   height
    '''
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array(([bbox[0], bbox[1], w, h]))


##################################################
# NEW: transform_cam_to_velo (camera -> velodyne)
##################################################
def transform_cam_to_velo(points_cam, calib_str):
    """
    points_cam: (N,3) array in camera coords
    calib_str:  multi-line KITTI calibration text
    returns (N,3) in velodyne coords
    """
    # 1) Build a TransformationKitti from the string
    t_kitti = TransformationKitti(calib_str, is_string=True)

    # 2) velodyne->camera matrix
    Tr_velo_to_cam = t_kitti.Tr_velo_to_cam_4x4  # shape (4,4)

    # 3) invert => camera->velodyne
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)

    # 4) apply transform
    N = points_cam.shape[0]
    hom_cam = np.hstack([points_cam, np.ones((N,1))])  # (N,4)
    hom_velo= hom_cam @ Tr_cam_to_velo.T               # (N,4)
    points_velo = hom_velo[:, :3]
    return points_velo

