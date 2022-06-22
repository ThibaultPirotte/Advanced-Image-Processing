# Some utility funcitons
# By Aurélie Bugeau
# June 2019

import torch
from torchvision import transforms
from PIL import Image
import numpy
from torch.autograd import Function
import matplotlib.pyplot as plt

#function to dispaly images with a title
def plotImage(image, legend, figsize_=(4,4)):
    plt.figure(figsize=figsize_)
    plt.imshow(image)
    plt.axis('off')
    plt.title(legend)
    plt.show()

# pre and post processing for images
prep = transforms.Compose([
        transforms.ToTensor(),
        #turn to BGR
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        #subtract imagenet mean
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(1.)),#255)),
        ])

postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1.)),#./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img



def get_input_noise(H, W, D):
    # filter noise with a gaussian to initialize synthesis
    if torch.cuda.is_available():
        gaussian_filt = torch.tensor([.1523,.2220,.2514,.2220,.1523],dtype=torch.float32,device=torch.device("cuda"))
    else:
        gaussian_filt = torch.tensor([.1523,.2220,.2514,.2220,.1523],dtype=torch.float32)

    gaussian_filt = gaussian_filt.view(1,-1)
    gaussian_filt = torch.mm(gaussian_filt.t(), gaussian_filt).view(1,1,5,5)

    if torch.cuda.is_available():
        input_img = torch.randn(D,1,H,W,dtype=torch.float32,device=torch.device("cuda"))
    else:
        input_img = torch.randn(D,1,H,W,dtype=torch.float32)

    for k in range(5):
        input_img = torch.nn.functional.conv2d(input_img,gaussian_filt,padding=2)
    noise_img = input_img.view(1,D,H,W)
    return noise_img

# Identity function that normalizes the gradient on the call of backwards
# Used for "gradient normalization"
class Normalize_gradients(Function):
    @staticmethod
    def forward(self, input):
        return input.clone()
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.mul(1./torch.norm(grad_input, p=1))
        return grad_input,
