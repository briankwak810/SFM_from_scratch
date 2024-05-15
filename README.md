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

# Results:

The results from the front and top are as follows. (For 14th output)

![Screenshot from 2024-05-10 23-55-46](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/fad08ea9-9909-46a6-8437-8be07cc6c6e0)

And the camera results are as follows. (For total 14 cameras)

![Screenshot from 2024-05-10 23-56-10](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/fb8dd5a4-3146-4584-882a-d931ea659be5)

For the results, I have observed that as the number of images used to reconstruct the 3d point cloud accumulates, outliers that are very far away sometimes appear. The output given below is an example of this, one of reconstructed “points_14” output, where the reconstructions for the main point cloud are very nice(the same as points_13) but because of one or two outliers the total point cloud is zoomed out so that the results do not look very convincing.

![Screenshot from 2024-05-11 00-05-01](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/32bfb0b8-583c-4bb2-8f54-278432475da9)

However, I was able to capture outputs without outliers, and they show very good reconstruction of the four pictures on the wall + some more. I believe that this is because of ransac randomness problems, because as points get larger the ransac error accumulates and makes **outliers far away that gives no problem whatsoever in the calculations, just because it is very far away(parallel light)**.

# Additional experiments:

The camera calibration was calculated with filming checkerboard with the same camera in different angles. The code is shown at calibration.ipynb.

The camera matrix is:

$$
\begin{bmatrix}
717.542 & 0 & 371.445 \\
0 & 714.882 & 501.326 \\
0 & 0 & 1
\end{bmatrix}
$$

By using the camera matrix, I was able to reconstruct two kinds of images, a tissue box with a hand and a tissue box in front of a bookcase. The first image set is in myimage_1 and the second image se t is in myimage_2 folder.

The feature matching + epipolar line visualizations are:

![Screenshot from 2024-05-10 23-45-24](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/633cd98d-4e4e-490b-8dea-e0b120610930)

The first example has a image of:



The reconstructed image was (the outputs are in myimages_1_output):

![Screenshot from 2024-05-10 21-49-39](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/2ead128e-6d49-4317-8ea6-e5e5620a0220)

The camera positions are:

![Screenshot from 2024-05-10 20-50-36](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/e8218de5-0397-4fb9-aa3b-35fe65a54920)

The second example has epipolar lines and feature matching of the first two images are:

![Screenshot from 2024-05-10 23-49-14](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/3b40be1c-3282-4d29-b9e9-80ae97fb7d09)

The reconstructed image was(the outputs are in myimage_2_output):

![Screenshot from 2024-05-10 23-50-56](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/9c6862e1-5534-493b-851a-1f19c6112d01)

It used 10 images to reconstruct this, and is shown in folder output_2. You can clearly see the feature points of the tissue, or the books in the shelf from the reconstruction.

The camera positions are:

![Screenshot from 2024-05-10 23-51-46](https://github.com/briankwak810/SFM_from_scratch/assets/119718552/ad3d1166-2748-42a3-873c-01de5bf5c076)


