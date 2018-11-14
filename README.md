# NEWS-phd
Notebooks and related info on the ML study of NEWSdm during PhD

The data is represented by 8 grey-scale images for each sample
* CNN - using 3D CNNs (8 images per sample stacked in a 3D image)
  * performance.ipynb - visualizing traning and test performance of the network on different signal classes.
  * explore_features.ipynb - exploring the data where network is certain in it's decision
  * old_vs_new.ipynb - comparing the performance of the CNN built and trained on the old dataset with the new residual CNN.
* RNN - using 8 images as a sequence for RNN (further plans).
* dataset.ipynb - just describes the way of converting images from csvs into hdf5 file.
* get_ims.C - script for extracting BFCL (best focused cluster) images from the root files.
