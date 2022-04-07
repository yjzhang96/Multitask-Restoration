import torch.utils.data as data
from PIL import Image
import os
import os.path
from glob import glob
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def _make_dataset(dataset_dirs,config):
    """
    Creates a 2D list of all the frames in N clips containing
    M frames each.

    2D List Structure:
    [[frame00, frame01,...frameM]  <-- clip0
     [frame00, frame01,...frameM]  <-- clip0
     :
     [frame00, frame01,...frameM]] <-- clipN

    Parameters
    ----------
        dir : string
            root directory containing clips.

    Returns
    -------
        list
            2D list described above.
    """


    # framesPath = []
    # # Find and loop over all the clips in root `dir`.
    # count = 0
    # for index, folder in enumerate(os.listdir(input_dir)):
    #     BlurryFolderPath = os.path.join(input_dir, folder)
    #     SharpFolderPath = os.path.join(target_dir, folder)

    #     # Skip items which are not folders.
    #     if not (os.path.isdir(BlurryFolderPath)):
    #         continue
    #     BlurryFramePath = sorted(os.listdir(BlurryFolderPath))
    #     for frame_index in range(len(BlurryFramePath)):
    #         framesPath.append({})
    #         framesPath[count]['INPUT'] = os.path.join(BlurryFolderPath,BlurryFramePath[frame_index])
    #         num_frame_B = int(BlurryFramePath[frame_index].split('.')[0])

    #         framesPath[count]['S'] = os.path.join(SharpFolderPath,"%06d.png"%(num_frame_B))
    #         count += 1
    # return framesPath
    
    
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    input_img_paths = []
    target_img_paths = []
    for dir_i in dataset_dirs:
        input_img_paths += sorted(glob(os.path.join(dir_i,'input/*'), recursive=True))
        target_img_paths += sorted(glob(os.path.join(dir_i,'target/*'), recursive=True))
    
    assert len(input_img_paths) == len(target_img_paths)
    
    count = 0
    for index in range(len(input_img_paths)):    
        framesPath.append({})
        framesPath[count]['input'] = input_img_paths[index]
        framesPath[count]['target'] = target_img_paths[index]
        count += 1
    return framesPath


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens image at `path` using pil and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, configional
            coordinates for cropping image. Default: None
        resizeDim : tuple, configional
            dimensions for resizing image. Default: None
        frameFlip : int, configional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            2D list described above.
    """


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = resized_img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')
    
    
class BlurryVideo(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:

        |-- video0
            |-- frameB0 frameB1 -- frameB0_S frameB0_E frameB1_S frameB1_E
            |-- frame01
            :
            |-- framexx
            |-- frame12
        |-- clip1
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
        :
        :
        |-- clipN
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12

    ...

    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.

    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """


    def __init__(self, config, train):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            transform : callable, configional
                A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            dim : tuple, configional
                Dimensions of images in dataset. Default: (640, 360)
            randomCropSize : tuple, configional
                Dimensions of random crop to be applied. Default: (352, 352)
            train : boolean, configional
                Specifies if the dataset is for training or testing/validation.
                `True` returns samples with data augmentation like random 
                flipping, random cropping, etc. while `False` returns the
                samples without randomization. Default: True
        """


        # Populate the list with image paths for all the
        # frame in `root`.
        self.config = config
        if train:
            dataset_dirs = [config['train']['train_dir_blur'],config['train']['train_dir_rain'],
                            config['train']['train_dir_noise']]
        else:
            if config['is_training']:
                ## validation
                dataset_dirs = [config['val']['val_dir_blur'],config['val']['val_dir_rain'],
                                config['val']['val_dir_noise']]
            else:
                ## test
                dataset_dirs = [config['test']['test_dir']]

        self.input_dir = dataset_dirs
        framesPath = _make_dataset(dataset_dirs,config)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: datasets"))
                
        self.patch_size = config['crop_size']
        
        self.framesPath     = framesPath
        self.train = train
        # mean = [0.5,0.5,0.5]
        # std = [1,1,1]
        # normalize = transforms.Normalize(mean=mean,
        #                                 std = std)
        self.transform = transforms.Compose([transforms.ToTensor() ])

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - B1 and B2 -
        and coresponding start and end frame groundtruth B1_S B1_E ... 

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            tuple
                (sample, returnIndex) where sample is 
                [I0, intermediate_frame, I1] and returnIndex is 
                the position of `random_intermediate_frame`. 
                e.g.- `returnIndex` of frame next to I0 would be 0 and
                frame before I1 would be 6.
        """


        sample = {}
        inp_path = self.framesPath[index]['input']
        tar_path = self.framesPath[index]['target']
        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        if (self.train):
            ps = self.patch_size
            ### Data Augmentation ###
            
            w,h = tar_img.size
            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0

            # Reflect Pad in case image is smaller than patch_size
            if padw!=0 or padh!=0:
                inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
                tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

            aug    = random.randint(0, 2)
            if aug == 1:
                inp_img = TF.adjust_gamma(inp_img, 1)
                tar_img = TF.adjust_gamma(tar_img, 1)

            aug    = random.randint(0, 2)
            if aug == 1:
                sat_factor = 1 + (0.2 - 0.4*np.random.rand())
                inp_img = TF.adjust_saturation(inp_img, sat_factor)
                tar_img = TF.adjust_saturation(tar_img, sat_factor)

            inp_img = TF.to_tensor(inp_img)
            tar_img = TF.to_tensor(tar_img)

            hh, ww = tar_img.shape[1], tar_img.shape[2]

            rr     = random.randint(0, hh-ps)
            cc     = random.randint(0, ww-ps)
            aug    = random.randint(0, 8)

            # Crop patch
            inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
            tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

            # Data Augmentations
            if aug==1:
                inp_img = inp_img.flip(1)
                tar_img = tar_img.flip(1)
            elif aug==2:
                inp_img = inp_img.flip(2)
                tar_img = tar_img.flip(2)
            elif aug==3:
                inp_img = torch.rot90(inp_img,dims=(1,2))
                tar_img = torch.rot90(tar_img,dims=(1,2))
            elif aug==4:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            elif aug==5:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            elif aug==6:
                inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            elif aug==7:
                inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
            
        
        else:
            if self.config['is_training']:
                # Validate on center crop
                if self.patch_size is not None:
                    ps = self.patch_size
                    inp_img = TF.center_crop(inp_img, (ps,ps))
                    tar_img = TF.center_crop(tar_img, (ps,ps))

                inp_img = TF.to_tensor(inp_img)
                tar_img = TF.to_tensor(tar_img)

            else:
                # test
                inp_img = TF.to_tensor(inp_img)
                tar_img = TF.to_tensor(tar_img)
        
        sample['input'] = inp_img
        sample['target'] = tar_img
        sample['B_path'] = self.framesPath[index]['input']
        sample['gt'] = True
        return sample


    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """


        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """
        for i in self.input_dir:
            input_dir = '{}\n'.format(i)
        print(input_dir)
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(input_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
