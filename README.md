# 3DSIFT PiDVC
3D SIFT aided Path-independent Digital Volume Correlation, a multi-thread CPU implementation.



Digital volume correlation is an non-invasive technique for measuring internal deformation between two volumetric images obtained before and after deformation. The DVC calculation runs on the two volume images, resulting the deformation vectors of points of interest (POIs) that are set on the reference image (image before deformation).

The computation of each POI is independent to other POIs in PiDVC . And the local image area around the POI (called as subvolume) is used in calculation. Hence the subvolume size should be set before calculation.



The DVC method aided by 3D SIFT is adaptive to large and complex deformation (*reference [3]*). Besides, the PiDVC method using FFT-CC is also provided (*reference  [1]*).  This work is based on our previous work on SIFT aided PiDIC, readers interested to the SIFT aided method can refer to the *reference [2]*.

This program is written in C++ language and parallelized using OpenMp.



Copyright (c) 2020 Communication and Computer Network Lab of Guangdong, more details in LICENSE (GPLV3).

## Contents

### DVC Algorithm

Two path-independent Digital volume (PiDVC) correlation methods are provided in this project:

1. 3D SIFT aided PiDVC
   - feature-based, adaptive to large and complex deformation
   - time-consuming
2. FFT-CC PiDVC
   - suitable to small deformation
   - fast

The users can choose one of the algorithms in the input configuration file, as seen in ***How to use***

### Components

The two initial guess methods and the 3D IC-GN algorithm are organized as classes, readers interested to the detailed implementation can refer to the source files.

- **3D SIFT aided estimation**
  - An initial guess method based on 3D SIFT matched keypoints, is capable to estimate sub-voxel level first-order deformation vector.
  - rely on *3D SIFT (imported as submodule)*, *Kd-Tree*, *Eigen*
  - source files:
    - `3DSIFT_PiDVC\Include\FitFormula.h`
    - `3DSIFT_PiDVC\Src\FitFormula.cpp`
    - The function of 3D SIFT extraction and matching is : `SIFT_PROCESS Sift_(const string Ref_file_name, const string Tar_file_name, CPaDVC* sepaDVC, SIFTMode Mode)`, in file `3DSIFT_PiDVC\Src\main.cpp`
- **FFT-CC estimation**
  - An initial guess method based on fast fourier transform (FFT) and inverse fast fourier transform (iFFT), is capable to estimate integer-voxel level displacements.
  - rely on *FFTW*.
  - source files:
    - `3DSIFT_PiDVC\Include\FFTCC.h`
    - `3DSIFT_PiDVC\Src\FFTCC.cpp`
- **3D IC-GN registration**
  - Iterative high-accuracy sub-voxel registration algorithm, fed with deformation vectors obtained by initial guess method.
  - Based on first-order shape function, and cubic Bspline interpolation
  - source files:
    - `3DSIFT_PiDVC\Include\ICGN.h`
    - `3DSIFT_PiDVC\Src\ICGN.cpp`



## How to use

**The DVC program is now running on command line, by passing the path of the configuration file. The settings of the calculation is read from the input configuration file (i.e. yml format).**

The guide consists of the following four parts.

###　Build the program from source

The steps of compiling and running the program is as follows:

1. clone the git repository 

2. init and update the submodule by commands

   - ```bash
     git submodule init
     git submodule update
     ```

3. Open the visual studio project `.sln` file.

4. Set the Windows SDK version of the visual studio project to that are installed on your PC. PS: version below  10.0.17763.0 is non-tested.

5. Build the release version of the program in visual studio

### Modify input configuration file

The configuration files are text files in *yml* format, several example configuration files are provided in path `Example/`.

Users might set the configuration file according to the following description

#### Essential fields

The following fields must be valid in calculation.

- `filepath` - `ref` - `image`
  -  path to the reference image (nifti format)
- `filepath` - `tar` - `image`
  -  path to the target image (nifti format)
- `filepath` - `result_dir`
  -  path to **directory** where the output text file is put inside
- `roi` - `mode`
  -  mode of ROI. This filed should be `cuboid` or `import`
- `roi` - `subset_radius`
  -  size of the subvolume (including x, y and z dimensions) *(integer values)*.
- `dvc` - `initial` - `method`
  -  method of initial guess. This field should be `PiSIFT` , `FFT-CC`  or `IOPreset`.
- `dvc` - `iterative` - `icgn_max_iter`
  -  the maximum iteration number of 3D IC-GN *(integer values)*.
- `dvc` - `iterative` - `icgn_deltaP`
  -  the convergence threshold of increment of displacement components *(floating point value)*.
- `dvc` - `num_thread`
  - number of threads *(integer value)*.

#### ROI settings

There are two kinds of ROI settings, users should set `roi` - `mode` as one of them .

- `cuboid`

  - A cuboid matrix of POIs are generated according to several fields:
  - required fields:
    - `roi` - `start_coor`, the start point of POI matrix (e.g. the POI closest to the (0,0,0))  *(integer values)*.
    - `roi` - `poi_num`, the dimensions of the POI matrix  *(integer values)*.
    - `roi` - `grid_space`, the space of adjacent POIs along different directions (x-axis, y-axis and z-axis) *(integer values)*.

- `import`

  - The coordinates of POIs are read from a text file

  - required fields:

    - `filepath` - `poi_coor`, the path to text file storing coordinates of POIs. 

  - In the text file, coordinate of each POI occupies a line, and the x,y and z coordinates are listed one by one and are separated by a comma.

  - Example:

    - ```
      50,60,70
      55,60,70
      ```

    - In this example, there are two POIs, the first is at (50,60,70) and the second is at (55,60,70). 

#### Initial estimation settings

There are three kinds of initial guess settings, users should set `dvc` - `initial` - `method` as one of them.

- `PiSIFT` 
  - 3D SIFT aided estimation
  - required fields:
    - `dvc` - `initial` - `ransac_error`, the error threshold in fitting affine matrix using RANSAC *(floating point value)*.
    - `dvc` - `initial` - `ransac_max_iter`, the maximum iteration number of RANSAC *(integer value)*.
    - `dvc` - `initial` - `min_neighbor`, the least number of nearby keypoints around a POI required for estimation *(integer value)*.
  
- `FFT-CC`
  - FFT-CC estimation
  - no extra required fields

- `IOPreset`

  - the initial value of each POI is obtained from a text file rather than using any algorithm to calculate

  - required:

    - the  `roi` - `mode` must be `import`
    - the initial value is after the coordinate of the POI in the text file set in `filepath` - `poi_coor`

  - Example:

    - ```
      50,60,70,1,2,3,-0.3,0.7,0,-0.7,-0.28,0,0,0,0
      55,60,70,2,1,3,-0.4,0.6,0,-0.6,-0.4,0,0,0,0
      ```

    - each POI occupies a line, and the initial values are `x,y,z,u,v,w,ux,uy,uz,vx,vy,vz,wx,wy,wz` , respectively

#### Details of fields 

The meanings of fields in the *yml* files are as follows:

```yml
filepath:
  ref:
    image: #path to reference image file 
    key_points: #path to coordinates of matched keypoints in reference image, only works when using 3D SIFT initial guess method
  tar:
    image: #path to target image file 
    key_points: #path to coordinates of matched keypoints in target image, only works when using 3D SIFT initial guess method
  result_dir: #path to the directory of results, the output text file will be automatically generated in the directory
  poi_coor: #
  sift_mode: #
roi:
  mode: cuboid
  start_coor: #the start coordinate of POI matrix
    x: 
    y: 
    z: 
  poi_num: #dimensions of POI matrix, i.e. x*y*z POIs totally
    x: 
    y: 
    z: 
  grid_space: #the grid space of adjacent POIs along different directions
    x: 
    y: 
    z: 
  subset_radius: #subvolume size 
    x: 
    y: 
    z: 
dvc:
  initial:
    method: PiSIFT
    ransac_error: #ransac error threshold, only works for PiSIFT initial method, recommended as 3.2
    ransac_max_iter: #ransac maximum iteration numbers, only works for PiSIFT initial method, recommend as 30
    min_neighbor: #minimum nearyby matched keypoints, only works for PiSIFT initial method
  iterative:
    icgn_max_iter: #maximum iteration number of 3D IC-GN
    icgn_deltaP: #the desired increment of displacement, once the increasement is smaller than the deltaP the algorithm stop (regarded as convergence). 0.001 is used in our work.
  num_thread: #number of CPU threads for parallel computing
```

### Run the program

The program receive a parameter, i.e. the path of the *yml* file.

Users should use command line, such as *cmd* or *bash* on windows.

```bash
3DSIFT_PiDVC.exe "a.yml"
```

First, users should change directory to where the program is. Then execute the program with the path of the specific *yml* file as parameter.

Screenshots of launching the program:

1. using *bash* on windows

![image-20200528234519400](.assert\image-20200528234519400.png)

2. using *cmd* on windows

![image-20200528235057629](.assert\image-20200528235057629.png)



> Besides, the program requires several *dll* files of the fftw3 library, and those *dll* files would be automatically copied to the *target directory* (i.e. the directory where the generated program is). If the users want to move the program to any other directory, please copy those *dll* files too.

### Results and analysis



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

[2] Yang, J., Huang, J., Jiang, Z., Dong, S., Tang, L., Liu, Y., ... & Zhou, L. (2020). SIFT-aided path-independent digital image correlation accelerated by parallel computing. *Optics and Lasers in Engineering*, *127*, 105964.

[3] Our paper in preparation.