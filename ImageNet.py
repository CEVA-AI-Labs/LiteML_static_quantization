
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from torchvision.models import alexnet

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision

TEST_DATA_FRAC = 0.01
CALIBRATION_SIZE = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageNetDataset(Dataset):
    def __init__(
        self,
        images_root,
        labels_path,
        classes_num=-1,
        transform=None,
        target_transform=None,
    ):
        """
        :param images_root: images folder to imagenet
        :param labels_path: labels folder for imagenet
        :param classes_num: number of classes to use
        :param transform: image transform
        :param target_transform: label transform
        """
        labelsToUse = None
        with open(labels_path) as lab:
            self.images = lab.readlines()
        self._labelsToUse = []
        if labelsToUse is not None and classes_num != -1:
            raise ValueError(
                " only one limiting parameter can be used, either class num or labelsToUse"
            )
        if classes_num != -1:
            allLabels = [int(image.split(" ")[1]) for image in self.images]
            allLabels = list(set(allLabels))
            labelsToUse = allLabels[:classes_num]
        self._labelsToUse = labelsToUse
        if self._labelsToUse is not None:
            self.images = [
                image
                for image in self.images
                if int(image.split(" ")[1]) in self._labelsToUse
            ]

        self.root = images_root
        self.images = self.images
        self.transform = transform
        self.target_transform = target_transform


    def getLabelsToUse(self):
        return self._labelsToUse

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        pickedGoodImage = False
        while not pickedGoodImage:
            image = self.images[index]
            name, label = image.split(" ")
            fullPath = os.path.join(self.root, name)
            if not os.path.exists(fullPath):
                index = index - 1 if index > 0 else index + 1
                print("image name: {} does not exist".format(fullPath))
                continue
            try:
                img = Image.open(fullPath)
            except:
                index = index - 1 if index > 0 else index + 1
                continue

            label = int(label)
            if not len(img.getbands()) == 3:
                index = index - 1 if index > 0 else index + 1
                continue
            try:
                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    label = self.target_transform(label)
            except:
                print(" transform failed on image {} ".format(fullPath))
                index = index - 1 if index > 0 else index + 1
                pickedGoodImage = False
                continue
            return (img, label)


def get_dataset(dataset_folder='/AI_Labs/datasets/ImageNet', batch_size=64):
    imagenet_labelsVal = f"{dataset_folder}/val-labels.txt"
    imagenet_folderVal = f"{dataset_folder}/val"
    resize_size, crop_size = 256, 224
    interpolation = InterpolationMode.BILINEAR
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = ImageNetDataset(
        images_root=imagenet_folderVal,
        labels_path=imagenet_labelsVal,
        transform=test_transform,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    calib = range(CALIBRATION_SIZE)
    calibration_dataset = torch.utils.data.Subset(test_dataset, calib)
    calibration_dataset.transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


    calibration_loader = torch.utils.data.DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    calibration_loader_key = lambda model, x: model(x[0].to(device))
    return val_loader, calibration_loader, calibration_loader_key