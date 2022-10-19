# SURE-MS-HS <br>
Pytorch codes for the paper "Hyperspectral super-resolution by unsupervised convolutional neural network and SURE", accepted in proceeding of IGARSS 2022, Kuala Lumpur, July 2022.<br>
**Authors:** Han V. Nguyen $^\ast \dagger$, Magnus O. Ulfarsson $^\ast$,  Johannes R. Sveinsson $^\ast$, and Mauro Dalla Mura $^\ddagger$ <br>
$^\ast$ Faculty of Electrical and Computer Engineering, University of Iceland, Reykjavik, Iceland<br>
$^\dagger$ Department of Electrical and Electronic Engineering, Nha Trang University, Khanh Hoa, Vietnam<br>
$^\ddagger$ GIPSA-Lab, Grenoble Institute of Technology, Saint Martin d’Hères, France.<br>
Email: hvn2@hi.is
<br>
<br>
 **Please cite our paper if you are interested**<br>
 @inproceedings{nguyen2022hyperspectral,
  title={Hyperspectral Super-Resolution by Unsupervised Convolutional Neural Network and Sure},
  author={Nguyen, Han V and Ulfarsson, Magnus O and Sveinsson, Johannes R and Dalla Mura, Mauro},
  booktitle={IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium},
  pages={903--906},
  year={2022},
  organization={IEEE}
}
## Abstract:<br>
Recent advances in deep learning (DL) reveal that the structure of a convolutional neural network (CNN) is a good image prior (called deep image prior (DIP)), bridging the model-based and DL-based methods in image restoration. However, optimizing a DIP-based CNN is prone to overfitting leading to a poorly reconstructed image. This paper derives a loss function based on Stein's unbiased risk estimate (SURE) for unsupervised training of a DIP-based CNN applied to the hyperspectral image (HSI) super-resolution. The SURE loss function is an unbiased estimate of the mean-square-error (MSE) between the clean low-resolution image and the low-resolution estimated image, which relies only on the observed low-resolution image. Experimental results on HSI show that the proposed method not only improves the performance, but also avoids overfitting.
<br>
## Usage:<br>
The following folders contanin:
- data: The simulated PU dataset.
- models: python scripts define the model (network structure)
- utils: additional functions<br>
**Run the jupyter notebook and see results.**
## Environment
- Pytorch 1.8
- Matplotlib
- Numpy, Scipy, Skimage.