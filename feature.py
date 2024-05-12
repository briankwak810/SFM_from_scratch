import cv2
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    # filter calculates 2 nearestNeighbors in des1
    filter = NearestNeighbors(n_neighbors=2)
    filter.fit(des1)
    match_dist = []
    match_index1 = []
    match_index2 = []
    for i in range(des2.shape[0]): # Check neighbor distance for element in des2
        dist, index = filter.kneighbors([des2[i]])
        if dist[0][0] < dist[0][1] * 0.8: # If the smallest distance is smaller than 80% of the second smallest distance
            match_dist.append(dist[0][0]) # append to distance, index list
            match_index1.append(index[0][0])
            match_index2.append(i)
    match_dist = np.asarray(match_dist)
    match_index1 = np.asarray(match_index1)
    match_index2 = np.asarray(match_index2)

    # filter2 calculates 2 nearestNeighbors in des2
    filter2 = NearestNeighbors(n_neighbors=2)
    filter2.fit(des2)
    delete = []
    for i in range(match_index1.size):
        dist, index = filter2.kneighbors([des1[match_index1[i]]]) # Biliniear check from des2
        if dist[0][0] > 0.8 * dist[0][1]: # If check does not pass, not a feature point!
            delete.append(i) # Therefore append to delete

    match_index1 = np.delete(match_index1, delete)
    match_index2 = np.delete(match_index2, delete)
    match_dist = np.delete(match_dist, delete)

    x1 = []
    x2 = []
    # For final saved indices -> append location (2d coordinate)
    for query, train in zip(match_index1, match_index2):
        x1.append([loc1[query].pt[0], loc1[query].pt[1]])
        x2.append([loc2[train].pt[0], loc2[train].pt[1]])
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    return x1, x2, match_index1



def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    
    n = x1.shape[0]
    A = np.zeros((n, 9)) # Use 8-point algorithm to extract E

    for i in range(n):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3) # Solution to Ax=0, where x is E in column vector form

    U, S, V = np.linalg.svd(E)
    S = np.array([1,1,0]) # Make E's singular values to [1, 1, 0] (rank 2)
    E = U @ np.diag(S) @ V # Recalculate E

    return E




def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    max_inliers = 0
    best_E = None
    best_inlier = None

    for i in range(ransac_n_iter):
        random_idx = random.sample(range(x1.shape[0]), 8)
        u = x1[random_idx]
        v = x2[random_idx]
        E = EstimateE(u, v) # Use 8-point DLT estimation

        # Calculate homogeneous coordinates of x1_h, x2_h
        x1_h = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_h = np.hstack((x2, np.ones((x2.shape[0], 1))))

        inlier_idx = []
        # If corresponding points and epipolar lines that are calculated are too far, that means that each point is an outlier
        for j in range(x1.shape[0]):
            line = E @ x1_h[j] # Calculate epipolar lines
            if np.absolute(np.dot(x2_h[j], line)) < ransac_thr:
                #/ np.sqrt(line[0]**2 + line[1]**2)
                inlier_idx.append(j)

        # Update max
        if len(inlier_idx) > max_inliers:
            max_inliers = len(inlier_idx)
            best_E = E
            best_inlier = inlier_idx

    return best_E, best_inlier



def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    
    n = Im.shape[0]
    Loc = []
    Des = []
    sift = cv2.SIFT_create()
    track = []

    for i in range(n):
        loc, des = sift.detectAndCompute(Im[i], None)
        Loc.append(loc)
        Des.append(des)
    
    for i in range(n):
        print(f"Calculating feature track till: {i}")
        track_i = np.full((n, len(Loc[i]), 2), -1, np.float32)
        for j in range(i+1, n):
            x1, x2, idx1 = MatchSIFT(Loc[i], Des[i], Loc[j], Des[j])
            x1_h = np.hstack((x1, np.ones((x1.shape[0], 1))))
            x2_h = np.hstack((x2, np.ones((x2.shape[0], 1))))
            
            # Homogeneous cood for normalized coordinates
            x1_n_h = x1_h @ np.linalg.inv(K).T
            x2_n_h = x2_h @ np.linalg.inv(K).T
            # 2D cood for normalized coordinates
            x1_n = x1_n_h[:, :2]
            x2_n = x2_n_h[:, :2]

            # Get inliers with ransac
            _, inliers = EstimateE_RANSAC(x1_n, x2_n, 200, 0.01)

            for inlier in inliers:
                # number of feature point = idx1[inlier]
                track_i[i][idx1[inlier]] = x1_n[inlier]
                track_i[j][idx1[inlier]] = x2_n[inlier]

        # check valid features that are not (-1, -1) and delete the rest
        valid_features = np.any(track_i != np.array(-1), axis=(0, 2))
        track_i = track_i[:, valid_features]
        # append to track
        track.append(track_i)

    track = np.concatenate(track, axis=1)

    return track