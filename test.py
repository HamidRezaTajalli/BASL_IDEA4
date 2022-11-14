from dataset_handler.cifar10 import get_dataloaders_backdoor

from models import get_model


dataloaders, classes_names = get_dataloaders_backdoor(2, 3, True, True, 1, 40)

model = get_model('stripnet', 'cifar10', 'client', 2)

for items in dataloaders['train'][0]:
    out = model(items[0])
    print(out.size())
    print(type(out))
