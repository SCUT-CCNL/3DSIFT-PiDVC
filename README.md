# 3DSIFT PiDVC
3D SIFT aided Path-independent Digital Volume Correlation, a multi-thread CPU implementation.



The DVC method aided by 3D SIFT is adaptive to large and complex deformation (reference [2]). Besides, the PiDVC method using FFT-CC is also provided (reference  [1]).

This program is written in C++ language and parallelized using OpenMp.



Copyright (c) 2020 Communication and Computer Network Lab of Guangdong, more details in LICENSE(GPLV3).

## Contents





## How to use

The steps of running the example data is as follows:

1. Open the visual studio project `.sln` file.

2. Set the Windows SDK version of the visual studio project to that are installed on your PC. PS: version below  10.0.17763.0 is non-tested.

3. Build and run the program under command line

   - ```bash
     $cd "pathToReleaseDir"
     $./3DSIFT_PiDVC.exe "pathToYml"
     ```



### Input



### Output



## Example data

Several nifti files are provided:

https://drive.google.com/open?id=1IlOBp1hu-KUh648lXR3YjgNih71Mj8ke



## Dependencies

The following libraries are used in this work, and are already included in this project files.

- [3D SIFT](https://github.com/ParallelCCNL/3DSIFT_mt), our  project of multi-thread 3D SIFT, imported as a submodule in `3DSIFT_PiDVC\3DSIFT`, to perform SIFT feature extraction and matching.
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), put in path `3DSIFT_PiDVC\3party\Eigen`, used to perform matrix  computation, such as fitting affine transform.
- [KdTree]( https://github.com/jtsiomb/kdtree ), put in path `3DSIFT_PiDVC\3party\kdTree`, used to accelerate the searching nearby keypoints (3D coordinates)
- [yaml-cpp]( https://github.com/jbeder/yaml-cpp ), put in path`3DSIFT_PiDVC\3party\yaml-cpp` , to parse the input yml file 
- [FFTW](http://www.fftw.org/), put in path `3DSIFT_PiDVC\3party\fftw`, to perform FFT and IFFT computation in FFT-CC method.



## Reference and Citations

If you want to cite this work, please refer to the papers as follows.

[1]  Wang, T., Jiang, Z., Kemao, Q., Lin, F., & Soon, S. H. (2016). GPU accelerated digital volume correlation. *Experimental Mechanics*, *56*(2), 297-309. 

[2] Our paper in preparation.