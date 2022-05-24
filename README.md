# SURE-MS-HS <br>
Pytorch codes for the paper "Hyperspectral super-resolution by unsupervised convolutional neural network and SURE"<br>
**Han V. Nguyen**$^\ast$, **Magnus O. Ulfarsson**$^\ast$,  **Johannes R. Sveinsson**$^\ast$, and **Mauro Dalla Mura**$^\dagger$ <br>
$^\ast$ Faculty of Electrical and Computer Engineering, University of Iceland, Reykjavik, Iceland<br>
$^\dagger$ GIPSA-Lab, Grenoble Institute of Technology, Saint Martin d’Hères, France.
<br>
<br>
## Abstract:<br>
Recent advances in deep learning (DL) reveal that the structure of a convolutional neural network (CNN) is a good image prior (called deep image prior (DIP)), bridging the model-based and DL-based methods in image restoration. However, optimizing a DIP-based CNN is prone to overfitting leading to a poorly reconstructed image. This paper derives a loss function based on Stein's unbiased risk estimate (SURE) for unsupervised training of a DIP-based CNN applied to the hyperspectral image (HSI) super-resolution. The SURE loss function is an unbiased estimate of the mean-square-error (MSE) between the clean low-resolution image and the low-resolution estimated image, which relies only on the observed low-resolution image. Experimental results on HSI show that the proposed method not only improves the performance, but also avoids overfitting.
<br>
## Usage:<br>
The following folders contanin:
- data: The simulated PU dataset.
- models: python scripts define the model (network structure)
- utils: additional functions<br>
Run the jupyter notebooks and see results.
## Environment
- Pytorch 1.8
- Numpy, Scipy, Skimage.