README
======

This dataset contains the 3D annotations of the Berkeley speech-to-gesture 
dataset originally prepared by Ginosar et al. (2019) with 2D face, body, 
and hand pose annotations.

The 3D annotations for the training, validation, and test data are stored in 
"$SPEAKER/train_$SPEAKER_data.npz", "$SPEAKER/dev_$SPEAKER_data.npz", and 
"$SPEAKER/test_$SPEAKER_data.npz" respectively in the Numpy NPZ file format where 
$SPEAKER is one of the six speaker names ("oliver", "seth", "chemistry", 
"ellen", "jon", and "conan").

For the train and validation data, each data point was prepared by extracting the 
relevant audio pose information over a video clip of 64 frames. Since the 
sampling rate is 15 fps, this corresponds to a video clip of about 4 seconds.
Each data point is obtained by slicing the original video in the dataset for 
every N frames where N is 5 for "oliver", 3 for "seth", and 1 for "chemistry",
"ellen", "jon", and "conan".

Unlike the training and validation set, the test sequences are 192 frames long
and there are no overlapping between them.

In addition to the dataset we used to train our model, we also provide another
version of the training and validation sets without any frame duplication.
The training and validation of this version can be found under 
"$SPEAKER/train_$SPEAKER_noduplication_data.npz" and 
"$SPEAKER/dev_$SPEAKER_noduplication_data.npz", respectively.
Both versions store the data in the same format.  

Each training and validation files consists of the following data:
- 'wav'     # shape: (n_points, n_wav_samples)
- 'body'    # shape: (n_points, n_frames, 165)
- 'face'    # shape: (n_points, n_frames, 257)
- 'imgs'    # shape: (n_points, n_frames,)
where "n_joints" is the number of data points (the number of sequences), 
"n_frames" is the number of frames for each sequence (64 frames for training
and validation, 192 frames for test sequences), "n_wav_samples" is the length 
of the WAV file for each sequence.

The 'body' variable corresponds to the 3D upper body and 3D hand pose positions
represented in meters. To obtain the XYZ coordinate, first reshape the numpy array 
into (n_points, n_frames, 3, n_joints).

The joints are labeled according to the following:
0 head_top
1 neck
2 right_shoulder
3 right_elbow
4 right_wrist
5 left_shoulder
6 left_elbow
7 left_wrist
8 right_hip
9 left_hip
10 pelvis
11 spine
12 head
13 right_hand_root
14 right_index_0
15 right_index_1
16 right_index_2
17 right_middle_0
18 right_middle_1
19 right_middle_2
20 right_little_0
21 right_little_1
22 right_little_2
23 right_ring_0
24 right_ring_1
25 right_ring_2
26 right_thumb_0
27 right_thumb_1
28 right_thumb_2
29 right_index_3
30 right_middle_3
31 right_little_3
32 right_ring_3
33 right_thumb_3
34 left_hand_root
35 left_index_0
36 left_index_1
37 left_index_2
38 left_middle_0
39 left_middle_1
40 left_middle_2
41 left_little_0
42 left_little_1
43 left_little_2
44 left_ring_0
45 left_ring_1
46 left_ring_2
47 left_thumb_0
48 left_thumb_1
49 left_thumb_2
50 left_index_3
51 left_middle_3
52 left_little_3
53 left_ring_3
54 left_thumb_3

Given such body joint labels, their parental relation is defined by the 
following list (the index of the list prepresents the child joint label):
PARENT = [1, 11, 1, 2, 3, 1, 5, 6, 10, 10, -1, 10, 1, 13, 13, 14, 15, 
        13, 17, 18, 13, 20, 21, 13, 23, 24, 13, 26, 27, 16, 19, 22, 25, 
        28, 34, 34, 35, 36, 34, 38, 39, 34, 41, 42, 34, 44, 45, 34, 47, 
        48, 37, 40, 43, 46, 49])
		
Please note that right_wrist and right_hand_root represent the same location
in the 3D space. This is also the case for left_wrist and left_hand_root.

80+80+3+3+64+27=257

The 'face' data corresponds to the 3DMM face model which consists of 80
face identity parameters, 80 face albedo parameters, 3 translation paramters,
3 rotation parameters, 64 face expression parameters, and 27 illumination
parameters, respectively. Please refer to the following paper for more information 
about the face model: 
https://gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf

The mesh template and PCA bases used to reconstruct the face mesh is stored in 
face_mesh/face_mesh_model.npz.

Please use the FaceMesh.py file in the face_mesh folder to reconstruct the full 
3D face mesh from any given face parameters and its corresponding PCA bases.

The 'wav' variable stores the audio files for each data points with the sampling
rate of 16000. The length of the data for each sequence is 68267 for both training 
and validation, and 204800 for the test sequences.

The 'imgs' variable stores the location of the RGB images for each frame in the
dataset. In order to use this data, you first need to download the Berkeley 
speech-to-gesture dataset which you can find in the following link:
https://github.com/amirbar/speech2gesture/blob/master/data/dataset.md

The 'imgs' data is stored as a string with the following format:
"/path/to/$SPEAKER/frames/$FILENAME" where $SPEAKER is the speaker name and
$FILENAME is the name of the image file.
To access the images, replace the string "/path/to/" with the directory you 
used to store the Berkeley dataset.

If you find this dataset useful, please cite our following paper:

@InProceedings{3dconvgesture_2021,
Author = {Habibie, Ikhsanul and Xu, Weipeng and Mehta, Dushyant and Liu, 
Lingjie and Seidel, Hans-Peter and Pons-Moll, Gerard and Elgharib, Mohamed 
and Theobalt, Christian},
Title = {Learning Speech-driven 3D Conversational Gestures from Video},
Booktitle = {ACM International Conference on Intelligent Virtual Agents (IVA)},
Year = {2021}
}

Please also cite the paper of Ginosar et al. (2019) who initiated the collection 
of the dataset on a large scale in-the-wild footages.

@InProceedings{ginosar2019gestures,
  author={S. Ginosar and A. Bar and G. Kohavi and C. Chan and A. Owens and J. Malik},
  title = {Learning Individual Styles of Conversational Gesture},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)}
  publisher = {IEEE},
  year={2019},
  month=jun
}