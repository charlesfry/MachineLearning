import torch
from torch.utils import data
import torchvision
from torchvision import transforms



# If running on Windows and you get a BrokenPipeError,
# try setting the num_worker of torch.utils.data.DataLoader() to 0.

transf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((.5,.5,.5),(.5,.5,.5))]
)

train_set = torchvision.datasets.CIFAR10(root='E:\Vision\CIFAR10',train=True,download=True)
data_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)

print(data_loader.batch_size)