import torch
import numpy as np
import matplotlib.pyplot as plt

def hwc2chw(x):
    return(x.permute(2,0,1))
def chw2hwc(x):
    return(x.permute(1,2,0))
def get_filter():
    h = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    return torch.from_numpy(h/np.sum(h))

def gaussian_filter(N=3, sigma=2.0):
    n = (N - 1) / 2.0
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.from_numpy(h)

def sreCal(Xref,X):
    '''Calculate signal to reconstructed error (SRE) between reference image and reconstructed image
    Input: Xref, X: reference and reconstructed images in shape [h,w,d]
    Output: aSRE average SRE in dB
            SRE_vec: SRE of each band'''
    mSRE=0
    if len(Xref.shape)==3:
        Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1],Xref.shape[2])
        X=X.reshape(X.shape[0]*X.shape[1],X.shape[2])
        SRE_vec=np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            SRE_vec[i]=10*np.log(np.sum(Xref[:,i]**2)/np.sum((Xref[:,i]-X[:,i])**2))/np.log(10)
        mSRE=np.mean(SRE_vec)
    else:
        pass
    return mSRE
def SNRCal(Xref,X):
    '''Calculate signal to reconstructed error (SRE) between reference image and reconstructed image
    Input: Xref, X: reference and reconstructed images in shape [h,w,d]
    Output: aSRE average SRE in dB
            SRE_vec: SRE of each band'''
    Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1]*Xref.shape[2])
    X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2])
    return  10*np.log(np.sum(Xref**2)/np.sum((Xref-X)**2))/np.log(10)

def compute_Hx(x,psf,ratio):
    '''Compute filtering and downsampling:
    x: torch image (NCHW)
    psf: torch PSF (NChw)
    ratio: downsampling ratio
    return: LR image y (NC/ratioW/ratioH)'''
    [N,D,nl,nc] = x.shape
    [d,dx,dy] = psf.shape
    B = torch.zeros_like(x)
    B[:,:, int(nl % 2 + (nl - dx) // 2):int(nl % 2 + (nl + dx) // 2),
                int(nc % 2 + (nc - dy) // 2):int(nc % 2 + (nc + dy) // 2)] = psf
    B = torch.fft.fftshift(B,dim=[2,3])
    # B=B/torch.sum(B)
    FBM = torch.fft.fft2(B)
    Xf = torch.real(torch.fft.ifft2(torch.fft.fft2(x) * FBM))
    return Xf[:,:,::ratio,::ratio]

def add_noise(x,sigma,device='cuda'):
    '''Add noise with std. sigma to image x (NCHW)'''
    xnoise = torch.zeros_like(x)
    for i in range(len(sigma)):
        xnoise[:,i,:,:] = x[:,i,:,:]+torch.randn(size=(x.shape[2],x.shape[3]),device=device)*sigma[i]
    return xnoise

def get_sigma(sigma=1,eta=20,band=93):
    '''Generate \sigma as a bell-curve function
    sigma_i**2 = sigma**exp(-(i-B/2)**2/(2*eta**2))/sum(exp(-(i-B/2)**2/(2*eta**2))'''
    den = 0
    num =[]
    for i in range(1,band+1):
        den = den + np.exp(-(i-band/2)**2/(2*eta**2))
    for k in range(band):
        num.append(np.exp(-(k-band/2)**2/(2*eta**2)))
    return np.sqrt(sigma**2*(num/den))
def add_noise(x,sigma,device='cuda'):
    '''Add noise with std. sigma to image x (NCHW)'''
    sigma = torch.from_numpy(sigma).to(device)
    xnoise = torch.zeros_like(x)
    for i in range(len(sigma)):
        xnoise[:,i,:,:] = x[:,i,:,:]+torch.randn(size=(x.shape[2],x.shape[3]),device=device)*sigma[i]
    return xnoise

def im2mat(X):
    """X(r,c,b)-->X(r*c,b)"""
    return X.reshape(X.shape[0]*X.shape[1],X.shape[2])

def mat2im(X,r):
    """X(r*c,b)-->X(r*c,b)"""
    c=int(X.shape[0]/r)
    b=X.shape[1]
    return X.reshape(r,c,b)
def hsi2msi(X,R):
    """Convert HSI X to MSI M by multiplying with spectral response R: M=X*R
    X: (HxWxNh), R(NhxNm)"""
    [r,c,b]=X.size()
    x=im2mat(X)
    if R.shape[0]!=b:
        R=torch.transpose(R,0,1)
    xout = torch.mm(x,R)
    return mat2im(xout,r)
##########################################################################
import torch
import numpy as np
def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.shape[0] - img.shape[0] % d,
                img.shape[1] - img.shape[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    if method == '2D':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    elif method == '3D':
        shape = [1, 1, input_depth, spatial_size[0], spatial_size[1]]
    else:
        assert False

    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False