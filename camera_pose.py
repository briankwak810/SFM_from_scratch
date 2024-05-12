import numpy as np

from feature import EstimateE_RANSAC


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    
    U, _, V = np.linalg.svd(E)
    u_3 = U[:,-1] # t
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_1 = U @ W @ V
    R_2 = U @ W.T @ V # R

    R_1 = R_1 if np.linalg.det(R_1) > 0 else -R_1
    R_2 = R_2 if np.linalg.det(R_2) > 0 else -R_2 # Check for det(R)=-1

    # Construct 4 possible camera coordinates
    R_set = np.array([R_1, R_1, R_2, R_2])
    C_set = np.array([-R_1.T @ u_3, R_1.T @ u_3, -R_2.T @ u_3, R_2.T @ u_3])

    return R_set, C_set



def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4):
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    X = []
    
    n = len(track1)
    for i in range(n):
        # Calculate triangulation only for matching feature points
        if track1[i, 0] == -1 or track2[i, 0] == -1:
            X.append(np.array([-1, -1, -1]))
        else:
            x1, y1 = track1[i]
            x2, y2 = track2[i]
            # Using DLT (2 points, 4 equations)
            A = np.array([x1 * P1[2,:] - P1[0,:], y1 * P1[2,:] - P1[1,:], x2 * P2[2,:] - P2[0,:], y2 * P2[2,:] - P2[1,:]])
            _, _, V = np.linalg.svd(A)
            v3 = V[-1]

            # Calculated 3d point from Homogeneous coordinate
            X.append(np.array([v3[0]/v3[3], v3[1]/v3[3], v3[2]/v3[3]]))

    X = np.asarray(X)

    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    n = X.shape[0]
    valid_index = np.zeros(n)    

    for i in range(n):
        X1_cam = np.dot(P1[:, :3], X[i]) + P1[:, 3]
        X2_cam = np.dot(P2[:, :3], X[i]) + P2[:, 3]

        # If points at camera coordinate have positive z value -> valid point for cheirality conditions
        if X1_cam[2] > 0 and X2_cam[2] > 0:
            valid_index[i] = 1

    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    
    valid = np.logical_and(np.all(track1 != -1, axis=1), np.all(track2 != -1, axis=1))
    x1 = track1[np.where(valid)]
    x2 = track2[np.where(valid)]

    # Use already normalized cood from track
    E, _ = EstimateE_RANSAC(x1, x2, 200, 0.01)

    R_set, C_set = GetCameraPoseFromE(E)
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    max_cnt = 0
    max_i = 0

    # Find maximum for cheirality count
    for i in range(4):
        P2 = np.concatenate((R_set[i], np.expand_dims(-R_set[i]@C_set[i], axis=1)), axis=1) # New projection matrix relative to P1
        X = Triangulation(P1, P2, track1, track2)
        chr = EvaluateCheirality(P1, P2, X)
        cnt = np.count_nonzero(chr)
        if cnt > max_cnt:
            max_cnt = cnt
            max_i = i
            max_X = X

    R = R_set[max_i]
    C = C_set[max_i]
    X = max_X

    return R, C, X