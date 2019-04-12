# NEWS-phd
Notebooks and related info on the ML study of NEWSdm during PhD

The data is represented by 9 grey-scale images for each sample (periodic boundary conditions)
* CNN - using 3D CNNs (9 images per sample stacked in a 3D image)
  * Conv4_3D_residual_v3.ipynb - CNN training with/without random rotations during training
  * performance_v3.ipynb - visualizing validation performance for networks trained on different signal-background
* 70nm - notebooks dedicated to 70nm crystal study
* RNN - using 9 images as a sequence for RNN (further plans).
* dataset.ipynb - loading images and features from csv's and ROOT, adding 9th polarisation as a copy of 1st, cleaning and saving to hdf5.
* get_ims.C - script for extracting BFCL (best focused cluster) images from the ROOT files.
* carb_test - checking correlations between barycenter shift analysis and CNNs for test carbon samples
