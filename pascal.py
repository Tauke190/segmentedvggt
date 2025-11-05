import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Define a transform (convert images to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load VOC Segmentation dataset (2007 or 2012)
dataset = torchvision.datasets.VOCSegmentation(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transform=transform,
    target_transform=transform
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Inspect one sample
image, mask = dataset[0]
print(image.shape, mask.shape)
