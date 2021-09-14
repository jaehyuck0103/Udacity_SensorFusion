Template from https://github.com/udacity/SFND_2D_Feature_Tracking

# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 



## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run specific mode with visualization: `./2D_feature_tracking [SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT] [BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT] [MAT_BF, MAT_FLANN] [SEL_NN, SEL_KNN] `
5. Run benchmark mode without visualization (It makes .csv files): `./2D_feature_tracking `



## Project Rubric

#### MP.1 Data Buffer Optimization:  Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end.

Before push a new element, pop the first element of `dataBuffer` if `dataBuffer.size() == dataBuffersize`. Shutdown program with error message if `dataBuffer.size() > dataBuffersize`, as it is impossible scenario.

#### MP.2 Keypoint Detection: Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.

I make wrapper function `detKeypoints()` to remove the duplicate codes for time measure and visualization. `detKeypoints()` launches one of `detKeypointShiTomasi()`, `detKeypointsHarris()`, and `detKeypointsModern()` according to the argument. I follow the config of the template code for `detKeypointsShiTomasi`, and the config and NMS of the lecture code for `detKeypointHarris()`. For the `detKeypointModeren()`, I use default settings of OpenCV. If an invalid `detectorType` comes, `detKeypointsModern()` raises an error message and shutdown the program.

#### MP.3 Keypoint Removal: Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

I use `std::remove_if()` with lambda function which checks a keypoint is out of `vehicleRect`.

#### MP.4 Keypoint Descriptors: Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.

`descKeypoints()` calculate a selected description. If an invalid `descriptorType` comes, `descKeypoints()` raises an error message and shutdown the program. I use default configs of OpenCV for the descriptor extractors.

#### MP.5 Descriptor Matching: Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.

`matchDescriptors()` select a matcher by string. For the BFMatcher, a proper `normType` is selected by `descriptType`. For the FlannBasedMatcher, `descSource` and `descRef` are casted to `CV_32F` to avoid bug.

#### MP.6 Descriptor Distance Ratio: Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

`matchDescriptors()` select NN matching or KNN matching. When KNN matching, the matches whoes first-best distance is lower than 0.8 x second-best distance survive.

#### MP.7 Performance Evaluation 1: Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

Check `keypoints.csv`.

#### MP.8 Performance Evaluation 2: Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

Check `matches.csv`.

Some invalid combinations are filtered out at the start of `runSeq()` and marked as nan in the csv file.

#### MP.9 Performance Evaluation 3: Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

Check `time.csv`.



##### Top3 Detector /Descriptor combinations

1. `FAST` / `BRIEF`
2. `FAST` / `ORB`
3. `FAST` / `BRISK`

The selected combinations show the fastest speed. They also make a resonable amount of matchings.
