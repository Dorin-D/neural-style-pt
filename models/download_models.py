# %% [code]
import torch
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url
import os


def main(location):
    os.makedirs(location, exist_ok=True)
    # Download the VGG-19 model and fix the layer names
    
    vgg19_location = path.join(location, "vgg19-d01eb7cb.pth")
    
    if (os.path.isdir(vgg19_location)):
        os.remove(vgg19_location)

    if not(os.path.exists(vgg19_location)):
        print("Downloading the VGG-19 model")
        sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth")
        map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
        torch.save(sd, path.join(location, "vgg19-d01eb7cb.pth"))

    vgg16_location = path.join(location, "vgg16-00b39a1b.pth")
    if (os.path.isdir(vgg16_location)):
        os.remove(vgg16_location)
    
    if not(os.path.exists(vgg16_location)):
        # Download the VGG-16 model and fix the layer names
        print("Downloading the VGG-16 model")
        sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth")
        map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
        torch.save(sd, path.join(location, "vgg16-00b39a1b.pth"))

    # Download the NIN model
    if False:
        #I
        print("Downloading the NIN model")
        if version_info[0] < 3:
            import urllib
            urllib.URLopener().retrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", path.join("models", "nin_imagenet.pth"))
        else: 
            import urllib.request
            urllib.request.urlretrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", path.join("models", "nin_imagenet.pth"))

    print("All models have been successfully downloaded")
    
    
if __name__ == "__main__":
    location = "models"
    #main(location)