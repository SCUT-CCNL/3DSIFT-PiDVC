# Nifti data 

The data is stored in compressed nifti format, and each `nii.gz` file contains only one volume image.

You may use Matlab to read or write such data.

## Simulated data

The simulated volume image, consisting of a cubic simulated object with many small hollow sphere inside. (Like foam meterial). The simulated object is applied with two kinds of deformation: rotation deformation and inhomogeneous deformation

- `Simulated_reference.nii.gz`:Reference volume, the volume image before deformation
- `Simulated_deformed_rotate_45degree.nii.gz`:Target volume, applied with 45-degree rotation deformation around the axis perpendicular to the center of xy-plane
- `Simulated_deformed_large_inhomogeneous.nii.gz`:Target volume, applied with inhomogeneous deformation along the z-axis



## Real test data

The real test data is from the free dataset of [Correlated Solution Inc](https://www.correlatedsolutions.com/)

This data is like applied with compression deformation along z-axis.

- `Torus_Ref.nii.gz`: Reference volume image
- `Torus_Def.nii.gz`: Deformed volume image



## Convert other file format to nii

Actually, we use matlab to generate a `nii.gz` file.

The command is as follows:

```matlab
#VolumeData is a 3D array in matlab, it is actually the volume data
niftiwrite(VolumeData,'Volume.nii','Compressed',true);
```

The readers may use matlab to read other formats into the memory and organize the data in 3D array, then use the `niftiwrite` command to generate a nii file.

It should be noticed that the data type of the 3D array in matlab affects the data type used in the nii file. The default `double` type in matlab might lead to a large space when writting nifti file.