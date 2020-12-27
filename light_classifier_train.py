import cv2
import pickle
import argparse
from os import mkdir
from tqdm import tqdm
from time import time
from os.path import join
from numpy import array as nparray
from numpy import asarray as npasarray
from numpy import transpose as nptranspose

from light_classifier import Light_Classifier
from light_classifier import binary_acc
from light_classifier import Light_Dataset
from light_classifier import Heavy_Classifier
from light_classifier import Ultra_Light_Classifier

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from torch.utils.data.dataloader import DataLoader

print(f"torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")
print(f"opencv version: {cv2.__version__}")

print("")

parser = argparse.ArgumentParser(description='Detect on CS:GO')
parser.add_argument('-i', help='string identifier for the model', type=str)
parser.add_argument('-e', help='number of epochs to be trained', type=int)
parser.add_argument('-b', help='batch_size', type=int)
parser.add_argument('-dp', help='path to dataset', type=str)
parser.add_argument('-sp', help='path to root directory where models will be saved', type=str)
parser.add_argument('-n', help='the shape (n x n), in pixels, of the inference image for light_classifier', type=int, nargs='?', default=32)
parser.add_argument('-s', help='torch random seed (for dataset splitting and shuffling)', type=int, nargs='?', default=42)
parser.add_argument('-dl', help='length of dataset', type=int, nargs='?', default=None)
parser.add_argument('-lr', help='weight_decay', type=int, nargs='?', default=0.003)
parser.add_argument('-wd', help='weight_decay', type=int, nargs='?', default=0)
args = parser.parse_args()

weights = args.w

SEED = args.s
torch.manual_seed(SEED)
model_number = args.i 
n_img_size = args. 
num_epochs = args.e
scale_factor = 1
batch_size = args.b 
dlength = args.dl 
dataset_path = args.dp 
model_save_path = args.sp 
lr = args.lr 
weight_decay = args.wd 

# SEED = 24
# torch.manual_seed(SEED)
# model_number = 999 #currently using '999' as "disposable" model_number :)
# n_img_size = 32
# num_epochs = 100
# scale_factor = 1
# batch_size = 16
# dlength = 2 # leave None for maximum dataset length
# dataset_path = "E:\\Documento\\outputs\\"  #remember to put "\\" at the end
# model_save_path = 'E:\\Documento\\output_nn\\'
# lr = 0.02 
# weight_decay = 0

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

# model = Ultra_Light_Classifier()
model = Light_Classifier()
# model = Heavy_Classifier()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

#model.apply(init_weights)
model = model.to(device)
print(model)

transform = transforms.Compose([
    #mean and std calculated from 100k dataset - see more in "findmeanstd.py file"
    transforms.Normalize(mean=[79.29117544808655, 70.79513926913451, 60.948315409534956], \
                        std=[50.36878945074385, 44.904560780963074, 39.40949111442829]),
    # transforms.Resize([int(IMG_SIZE_X*scale_factor), int(IMG_SIZE_Y*scale_factor)]),
    transforms.ToTensor() 
])

#load dataset
dataset = Light_Dataset(dataset_path, transform=transform, img_size=n_img_size, dlength=dlength)

# a simple custom collate function, just to show the idea def my_collate(batch):
def my_collate_2(batch):
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return [imgs, labels]


train_set, val_set, _ = dataset.split(train=0.7, val=0.15, seed=SEED)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)


# criterion = CrossEntropyLoss()
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
log_interval = len(train_loader) // 1
log_interval_val = len(val_loader) // 1

print(f"Started training! Go have a coffee/mate/glass of water...")
print(f"Log interval: {log_interval}")
print(f"Please wait for first logging of the training")


def train_cycle():
    #safety toggle to make sure no files are overwritten by accident while testing!
    if model_number != 999:
        safety_toggle = input(f'ATTENTION: MODEL NUMBER IS :{model_number}:\
            ANY FILES WITH THE SAME MODEL NUMBER WILL BE DELETED. Continue? (Y/n):')
        if safety_toggle != 'Y' and safety_toggle != 'y':
            raise ValueError('Please change the model number to 999, or choose to continue')

    loss_total_dict = { 
        'epochs' : 0,
        'lr' : lr,
        'dset_size' : len(dataset),
        'weight_decay' : weight_decay,
        'seed' : SEED,
        'losses' : [],
        'accuracies' : [],
        'losses_val' : [],
        'accuracies_val' : [],
        'best_loss' : 999,
        'best_loss_val' : 999
    }

    model_save_path_new = join(model_save_path, str(model_number))
    mkdir(model_save_path_new)


    for epoch in range(num_epochs):  # loop over the dataset multiple times
        tic = time()

        running_loss = 0.0
        running_loss_val = 0.0
        running_acc = 0.0
        running_acc_val = 0.0

        ################## TRAINING STARTS ######################## 
        model.train()
        print(f'training epoch #{epoch+1}:')
        for i, data in enumerate(tqdm(train_loader, leave=False)):
            imgs, labels = data
            # img = imgs[0].numpy().copy().transpose(1, 2, 0)
            # cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            imgs = npasarray(list(nparray(im) for im in imgs))
            imgs = nptranspose(imgs, (0, 3, 1, 2))
            imgs = torch.from_numpy(imgs).float().to(device)
            
            #transform list of 0 dim tensors into one 1 dim tensor
            labels_cat = torch.tensor([label.item() for label in labels]).float().reshape((-1, 1))
            labels_cat = labels_cat.to(device)

            optimizer.zero_grad()

            preds = model(imgs)
            # print(preds)
            # print(preds.shape)
            # print(labels)
            # print(labels_cat.shape)
            loss = criterion(preds, labels_cat)
            acc = binary_acc(preds, labels_cat)

            running_loss += loss.item() 
            running_acc += acc.item()

            if (i + 1) % log_interval == 0:
                print('%s ::Training:: [%d, %5d] loss: %.5f' %
                    (model_number, epoch + 1, i + 1, running_loss / log_interval))

                loss_total_dict['epochs'] = epoch 
                loss_total_dict['losses'].append(running_loss/log_interval) 
                loss_total_dict['accuracies'].append(running_acc/log_interval) 

                #save model
                if running_loss/log_interval <= loss_total_dict['best_loss']:
                    loss_total_dict['best_loss'] = running_loss/log_interval
                    torch.save(model.state_dict(), join(model_save_path_new, "best_train") + '.th')
                else:
                    torch.save(model.state_dict(), join(model_save_path_new, "last") + '.th')
                    
                with open(f'{model_save_path_new}-train', 'wb') as filezin:
                    pickle.dump(loss_total_dict, filezin)

                running_loss = 0.0
                running_acc = 0.0

            loss.backward()
            optimizer.step()

        ################## VALIDATION STARTS ######################## 
        model.eval()
        print(f'validating epoch #{epoch+1}:')
        for i, data in enumerate(tqdm(val_loader, leave=False)):
            imgs, labels = data

            imgs = npasarray(list(nparray(im) for im in imgs))
            imgs = nptranspose(imgs, (0, 3, 1, 2))
            imgs = torch.from_numpy(imgs).float().to(device)
            
            #transform list of 0 dim tensors into one 1 dim tensor
            labels_cat = torch.tensor([label.item() for label in labels]).float().reshape((-1, 1))
            labels_cat = labels_cat.to(device)

            #running model
            preds = model(imgs)

            loss = criterion(preds, labels_cat)
            acc = binary_acc(preds, labels_cat)

            running_loss_val += loss.item() 
            running_acc_val += acc.item()

            if (i + 1) % log_interval_val == 0:
                print('%s ::Validation:: [%d, %5d] loss: %.5f' %
                    (model_number, epoch + 1, i + 1, running_loss_val / log_interval_val))
                print(f'Taking (precisely) {(time()-tic)/60} minutes per epoch')

                loss_total_dict['epochs'] = epoch
                loss_total_dict['losses_val'].append(running_loss_val/log_interval_val) 
                loss_total_dict['accuracies_val'].append(running_acc_val/log_interval_val) 

                if running_loss/log_interval <= loss_total_dict['best_loss_val']:
                    loss_total_dict['best_loss_val'] = running_loss_val/log_interval_val
                    torch.save(model.state_dict(), join(model_save_path_new, "best_val") + '.th')

                if epoch in checkpoints: 
                    with open(join(model_save_path_new, str(model_number))+"-train", 'wb') as filezin:
                        pickle.dump(loss_total_dict, filezin)

                running_loss_val = 0.0
                running_acc_val = 0.0

        print("\n")

train_cycle()