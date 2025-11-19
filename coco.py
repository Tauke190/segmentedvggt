from torchvision.datasets import CocoSegmentation
from torch.utils.data import DataLoader

# Paths to your COCO images and annotation files
train_img_dir = "dataset/train2017"
train_ann_file = "dataset/annotations/instances_train2017.json"

def coco_transform(image, target):
    image = torchvision.transforms.ToTensor()(image)
    target = torchvision.transforms.PILToTensor()(target).squeeze(0).long()
    return image, target

# Create the dataset
train_dataset = CocoSegmentation(
    root=train_img_dir,
    annFile=train_ann_file,
    transforms=coco_transform
)

# Create the dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)