# Prerequisites:

Import random in pnp.py

import random/sklearn.neighbors in feature.py

Look at 'requirements.txt' for more requirements.

# Algorithm:

All the coordinates are normalized with the intrinsic parameter.

The coordinates are called in EstimateE_RANSAC function, where the function estimates the best essential matrix fit from 8-point DLT algorithm. The threshold is calculated with the inner product that denotes the distance from the epipolar line to the corresponding point.

For example, the calculated epipolar line using the relation F=K^-T F K^-1 is shown for the given images 1 and 2. The visualization code is given in visualize_epipolar.py.

![Screenshot from 2024-05-10 23-53-25](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/0215081a-df69-4578-b820-c3107e9d6897)

(I have understood that the epipolar lines are not completely same, due to ransac randomness and threshold choices. Actually, the OpenCV code reconstructs epipolar lines differently too when reloaded different times.)

In the BuildFeatureTrack function, the whole track is built incrementally using the EstimateE_RANSAC function. The ransac algorithm outputs inliers, that is used in building matching feature tracks.

For PnP algorithms, DLT is used to calculate R and t from the matching 2d and 3d points. Because there are a total of 11 unknowns, we have to have at least 6 corresponding points and that is implemented as the minimum number of random indices in PnP_RANSAC function. In the ransac algorithm, the threshold is calculated by the reprojection error, or the 2d distance on the focal plane.

All the functions in camer_pose.py is the same as in-class implementations, where the cheirality is evaluated for 4 candidates of the pose matrix.

The results from the front and top are as follows. (For 14th output)

![Screenshot from 2024-05-10 23-55-46](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/fad08ea9-9909-46a6-8437-8be07cc6c6e0)

![Screenshot from 2024-05-10 23-51-46](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/e18ff85e-642f-4248-97db-89c6f26600cf)

And the camera results are as follows. (For total 14 cameras)

![Screenshot from 2024-05-10 23-56-10](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/fb8dd5a4-3146-4584-882a-d931ea659be5)

For the results, I have observed that as the number of images used to reconstruct the 3d point cloud accumulates, outliers that are very far away sometimes appear. The output given below is an example of this, one of reconstructed “points_14” output, where the reconstructions for the main point cloud are very nice(the same as points_13) but because of one or two outliers the total point cloud is zoomed out so that the results do not look very convincing.

![Screenshot from 2024-05-11 00-05-01](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/32bfb0b8-583c-4bb2-8f54-278432475da9)

However, I was able to capture outputs without outliers, and they show very good reconstruction of the four pictures on the wall + some more. I believe that this is because of ransac randomness problems, because as points get larger the ransac error accumulates and makes **outliers far away that gives no problem whatsoever in the calculations, just because it is very far away(parallel light)**.
