# NEWS-phd
Notebooks and related info on the ML study of NEWSdm during PhD

The data is represented by 8 grey-scale images for each sample
* CNN - using 3D CNNs (8 images per sample stacked in a 3D image)
  * Conv4_3D_residual_v2.ipynb - CNN training with 6-folds validation
  * performance.ipynb - visualizing traning and test performance of the network on different signal classes.
  * performance.ipynb - visualizing validation performance for 6 networks
  * explore_features.ipynb - exploring the data where network is certain in it's decision
  * old_vs_new.ipynb - comparing the performance of the CNN built and trained on the old dataset with the new residual CNN.
* RNN - using 8 images as a sequence for RNN (further plans).
* dataset.ipynb - loading images and features from csv's and ROOT, cleaning and saving to hdf5.
* get_ims.C - script for extracting BFCL (best focused cluster) images from the ROOT files.
