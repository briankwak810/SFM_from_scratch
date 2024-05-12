import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment


if __name__ == '__main__':
    np.random.seed(100)
    K = np.asarray([
        [463.1, 0, 333.2],
        [0, 463.1, 187.5],
        [0, 0, 1]
    ])
    num_images = 14
    w_im = 672
    h_im = 378

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = 'images/image{:d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)

    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))
    # Set first two camera poses
    P[0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    t = -R @ C
    P[1] = np.hstack((R, t.reshape(3, 1)))

    ransac_n_iter = 200
    ransac_thr = 0.01
    for i in range(2, num_images):
        # Estimate new camera pose
        valid = (X[:, 0] != -1) & (track[i, :, 0] != -1) # Get indices where point is constructed & is in image i
        X_valid = X[valid, :]
        x_valid = track[i, valid, :]

        R_ransac, C_ransac, inliers = PnP_RANSAC(X_valid, x_valid, ransac_n_iter, ransac_thr) # Run ransac with valid indices
        R, C = PnP_nl(R_ransac, C_ransac, X_valid[inliers.astype(bool)], x_valid[inliers.astype(bool)]) # Run refinement with inliers

        # Add new camera pose to the set
        t = -R @ C
        P[i] = np.hstack((R, t.reshape(3, 1))) # add P[i]

        for j in range(i):
            # Find new points to reconstruct
            new_point = FindMissingReconstruction(X, track[i,:,:])

            # Triangulate points
            X_new = Triangulation(P[j], P[i], track[j], track[i])
            valid = X_new[:, 0] != -1
            
            # X_refined_new denotes points in X_new that are not -1 and is refined with Triangulation_nl
            X_refined_new = Triangulation_nl(X_new[valid,:], P[j], P[i], track[j,valid,:], track[i,valid,:])
            X_new[valid,:] = X_refined_new

            # Filter out points based on cheirality
            cheir_valid_idx = EvaluateCheirality(P[j], P[i], X_refined_new)
            
            valid_pos = np.asarray(np.where(valid)[0])
            X_new[valid_pos[~cheir_valid_idx.astype(bool)]] = np.array([-1, -1, -1])

            # Update 3D points
            new_point_valid = np.asarray(np.where(new_point))[0]
            for k in range(X.shape[0]):
                if new_point[k] == 1:
                    X[k] = X_new[k]
        
        # Run bundle adjustment
        valid_ind = X[:, 0] != -1

        print(f"valid_ind size: {np.shape(np.where(valid_ind))}")

        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)

        P[:i+1,:,:] = P_new
        X[valid_ind,:] = X_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)