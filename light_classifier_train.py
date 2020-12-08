import pickle
import cv2
from tqdm import tqdm
from time import time
from numpy import array as nparray
from numpy import asarray as npasarray
from numpy import transpose as nptranspose

from light_classifier import Light_Classifier
from light_classifier import binary_acc
from light_classifier import Light_Dataset

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

print(f"torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")
print(f"opencv version: {cv2.__version__}")

print("")

SEED = 42
IMG_SIZE_X = 1280
IMG_SIZE_Y = 720

torch.manual_seed(SEED)
n_img_size = 28
num_epochs = 200
scale_factor = 1
# checkpoints = [14, 19, 49, 79, 99, 119, 149, 179, 199] #all epoch indexes where the network should be saved
checkpoints = [0, 14, 19, 49, 79, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
model_number = 999 #currently using '999' as "disposable" model_number :)
batch_size = 1
convs_backbone = 1
out_channels_backbone = 4
reg_weight = 1 # leave 1 for no weighting

# dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"  #remember to put "/" at the end
dataset_path = "E:\\Documento\\outputs\\"  #remember to put "\\" at the end
# dataset_path = "/home/sequoia/data/"
model_save_path = 'E:\\Documento\\output_nn\\'

#OPTIMIZER PARAMETERS ###############
lr = 1
weight_decay = 0

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

# model = Light_Classifier(num_convs_backbone=convs_backbone, num_backbone_out_channels=out_channels_backbone)
model = Light_Classifier()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

#model.apply(init_weights)
model = model.to(device)
print(model)

transform = transforms.Compose([
    transforms.Resize([int(IMG_SIZE_X*scale_factor), int(IMG_SIZE_Y*scale_factor)]),
    transforms.ToTensor(), # will put the image range between 0 and 1
])

#load dataset
dataset = Light_Dataset(dataset_path, transform=transform, img_size=n_img_size, dlength=1200)

# a simple custom collate function, just to show the idea def my_collate(batch):
def my_collate_2(batch):
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return [imgs, labels]


train_set, val_set, _ = dataset.split(train=0.7, val=0.15, seed=SEED)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)


criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
log_interval = len(train_loader) // 1
log_interval_val = len(val_loader) // 1

print(f"Started training! Go have a coffee/mate/glass of water...")
print(f"Log interval: {log_interval}")
print(f"Please wait for first logging of the training")

#safety toggle to make sure no files are overwritten by accident while testing!
if model_number != 999:
    safety_toggle = input('ATTENTION: MODEL NUMBER IS :{model_number}:\
        ANY FILES WITH THE SAME MODEL NUMBER WILL BE DELETED. Continue? (Y/n):')
    if safety_toggle != 'Y' and safety_toggle != 'y':
        raise ValueError('Please change the model number to 999, or choose to continue')


def train_cycle():
    loss_total_dict = { 
        'num_epochs' : num_epochs,
        'lr' : lr,
        'weight_decay' : weight_decay,
        'seed' : SEED,
        'losses' : [],
        'accuracies' : [],
        'losses_val' : [],
        'accuracies_val' : []
    }

    model_save_path_new = f"{model_save_path}model#{model_number}"  

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        tic = time()

        running_loss = 0.0
        running_loss_val = 0.0
        running_acc = 0.0
        running_acc_val = 0.0

        ################## TRAINING STARTS ######################## 
        model.train()
        print(f'training epoch #{epoch}:')
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
            labels_cat = torch.tensor([])
            for label in labels:
                labels_cat = torch.cat((label.view(1), labels_cat))
            labels_cat = labels_cat.long().to(device)

            optimizer.zero_grad()

            preds = model(imgs)

            loss = criterion(preds, labels_cat)
            loss_value = loss.item()

            acc = binary_acc(preds, labels_cat)

            running_loss += loss_value 
            running_acc += acc

            if (i + 1) % log_interval == 0:
                print('%s ::Training:: [%d, %5d] loss: %.5f' %
                    (model_number, epoch + 1, i + 1, running_loss / log_interval))

                loss_total_dict['losses'].append(running_loss) 
                loss_total_dict['accuracies'].append(running_acc) 
                running_loss = 0.0
                running_acc = 0.0

                if epoch in checkpoints: 
                    print(f"Saving net at: {model_save_path_new}")
                    torch.save(model.state_dict(), model_save_path_new + 'e' + f'{epoch}' + '.th')
                    
                with open(f'{model_save_path_new}-train', 'wb') as filezin:
                    pickle.dump(loss_total_dict, filezin)

            loss.backward()
            optimizer.step()
        print ("\033[A                             \033[A")

        ################## VALIDATION STARTS ######################## 
        model.eval()
        print(f'validating epoch #{epoch}:')
        for i, data in enumerate(tqdm(val_loader, leave=False)):
            imgs, labels = data

            imgs = npasarray(list(nparray(im) for im in imgs))
            imgs = nptranspose(imgs, (0, 3, 1, 2))
            imgs = torch.from_numpy(imgs).float().to(device)
            
            #transform list of 0 dim tensors into one 1 dim tensor
            labels_cat = torch.tensor([])
            for label in labels:
                labels_cat = torch.cat((label.view(1), labels_cat))
            labels_cat = labels_cat.long().to(device)

            #running model
            preds = model(imgs)

            loss = criterion(preds, labels_cat)
            acc = binary_acc(preds, labels_cat)

            running_loss_val += loss_value 
            running_acc_val += acc

            # #forward prop
            # bboxes_pred, pred_cls, pred_scores = fastercnn.get_prediction_fastercnn(
            #     imgs[0].to(device), net, threshold, category_names=categories, img_is_path=False)
            # cls_gt = np.array(categories)[[t.item() for t in targets[0]]]

            if (i + 1) % log_interval_val == 0:
                print('%s ::Validation:: [%d, %5d] loss: %.5f' %
                    (model_number, epoch + 1, i + 1, running_loss_val / log_interval_val))
                print(f'Taking (precisely) {(time()-tic)/60} minutes per epoch')

                loss_total_dict['losses_val'].append(running_loss_val) 
                loss_total_dict['accuracies_val'].append(running_acc_val) 
                running_loss_val = 0.0
                running_acc_val = 0.0

                if epoch in checkpoints: 
                    with open(f'{model_save_path_new}-train', 'wb') as filezin:
                        pickle.dump(loss_total_dict, filezin)
        print ("\033[A                             \033[A")

train_cycle()