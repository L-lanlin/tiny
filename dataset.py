import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os.path import join
import numpy as np
import os.path as osp
import sys
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


# Transforms for low resolution images and high resolution images
def transform_hl_pair(hr_height, hr_width):

    lr_transforms = [transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    hr_transforms = [transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(lr_transforms), transforms.Compose(hr_transforms)

def arrange_data(path):

    with open(path, 'r') as f:
        data = f.readlines()

    data = [x.strip() for x in data]
    flags = []
    for (i, x) in enumerate(data):
        if (x.endswith('.jpg')):
            flags.append(i)
        else:
            data[i] = [int(loc) for loc in x.split(' ')[:4]]

    path = np.array(data)[flags].tolist()
    bbxs = [x[2:] for x in np.split(data, flags[1:])]
    return path, bbxs

def iou(a, b):
    sizea = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    sizeb = (b[2] - b[0]) * (b[3] - b[1])
    tl = np.maximum(a[:, :2], b[:2])
    br = np.minimum(a[:, 2:], b[2:])
    wh = np.maximum(br - tl, 0)
    size = wh[:, 0] * wh[:, 1]
    return size / (sizea + sizeb - size)
    
# VOC_CLASSES = (  # always index 0
#     '1', '2', '3', '4',
#     '5', '6', '7', '8', '9',
#     '10', '11', '12', '13',
#     '14', '15', '16',
#     '17', '18', '19', '20')

VOC_CLASSES = ( # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # label_idx = self.class_to_ind[name]
            # bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'),('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712',high_resolution=(128, 128)):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.lr, self.hr = transform_hl_pair(*high_resolution)
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        img = self.pull_item(index)

        return img

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        # img = cv2.imread(self._imgpath % img_id)
        img = Image.open(self._imgpath % img_id)
        # height, width, channels = img.shape
        height, width = img.size

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # random select one face
        # idx = np.random.randint(0, len(target), 1)
        target = np.vstack(target)

        # target[:, 2:] += target[:, 0:2]

        # bbx = target[idx, :].squeeze()
        bbx = target[np.random.randint(len(target))].squeeze()
        true = img.crop(bbx)
        # random crop a fix-sized background patch
        x, y = np.random.randint(0, min(img.size) - 128, 2)
        bg = [x, y, x + 128, y + 128]
        if np.all(iou(target, bg) < 0.5):
            false = img.crop(bg)
        else:
            false = Image.fromarray(np.random.randint(0, 256, size=(128, 128, 3)).astype('uint8'))
            print("use random noise.")
        return {"lr_face": self.lr(true), "lr_background": self.lr(false),
                "hr_face": self.hr(true), "hr_background": self.hr(false)}

        # return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


class WIDER(Dataset):

    def __init__(self, base, path, bbxs, high_resolution=(128, 128)):
        self.base = base
        self.path = path
        self.bbxs = bbxs
        self.lr, self.hr = transform_hl_pair(*high_resolution)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img = Image.open(join(self.base, self.path[idx]))
        bbxs = np.vstack(self.bbxs[idx])
        # random select one face
        idx = np.random.randint(0, len(bbxs), 1)
        bbxs[:, 2:] += bbxs[:, 0:2]

        bbx = bbxs[idx, :].squeeze()
        true = img.crop(bbx)
        # random crop a fix-sized background patch
        x, y = np.random.randint(0, min(img.size) - 128, 2)
        bg = [x, y, x + 128, y + 128]
        if np.all(iou(bbxs, bg) < 0.5):
            false = img.crop(bg)
        else:
            false = Image.fromarray(np.random.randint(0, 256, size=(128, 128, 3)).astype('uint8'))
            print("use random noise.")
        return {"lr_face": self.lr(true), "lr_background": self.lr(false),
                "hr_face": self.hr(true), "hr_background": self.hr(false)}


if __name__ == '__main__':
    train_path = "./WIDER/WIDER_train/images/"
    path, bbxs = arrange_data()
    wider = WIDER(train_path, path, bbxs)
    result = wider[22]