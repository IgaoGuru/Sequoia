from light_classifier import Light_Classifier
from light_classifier import binary_acc

import torch
import torchvision
from torchvision import transforms

print(f"torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")
print(f"opencv version: {cv2.__version__}")

print("")

SEED = 42
IMG_SIZE_X = 1280
IMG_SIZE_Y = 720

torch.manual_seed(SEED)
scale_factor = 0.5
num_epochs = 200
# checkpoints = [14, 19, 49, 79, 99, 119, 149, 179, 199] #all epoch indexes where the network should be saved
checkpoints = [14, 19, 49, 79, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
model_number = 1 #currently using '999' as "disposable" model_number :)
batch_size = 1
convs_backbone = 1
out_channels_backbone = 4
reg_weight = 1 # leave 1 for no weighting

# dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"  #remember to put "/" at the end
dataset_path = "E:\\Documento\\outputs\\"  #remember to put "/" at the end
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

#dataset = CsgoPersonFastRCNNDataset(dataset_path, transform)
dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform, scale_factor=scale_factor, dlength=None)

# a simple custom collate function, just to show the idea def my_collate(batch):
def my_collate(batch):
    imgs = [item[0] for item in batch]
    targets = [(item[1][0], item[1][1]) for item in batch]
    img_paths = [item[2] for item in batch]
    return [imgs, targets, img_paths]

def my_collate_2(batch):
    imgs = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return [imgs, bboxes, labels]


train_set, val_set, _ = dataset.split(train=0.7, val=0.15, seed=SEED)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)


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
        'loss_sum' : [],
        'loss_classifier' : [],
        'loss_box_reg' : [],
        'loss_objectness' : [],
        'loss_rpn_box_reg' : []
    }

    loss_total_dict_val = { 
        'loss_sum' : [],
        'loss_classifier' : [],
        'loss_box_reg' : [],
        'loss_objectness' : [],
        'loss_rpn_box_reg' : []
    }

    model_save_path_new = f"{model_save_path}model#{model_number}"  

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        tic = time()

    #BUG: NO IDEA WHY: BONDING WITH UTILS.PY loss_per_epoch = {
        loss_per_epoch = {
            'loss_sum' : [],
            'loss_classifier' : [],
            'loss_box_reg' : [],
            'loss_objectness' : [],
            'loss_rpn_box_reg' : [] }
        running_loss = 0.0

        loss_per_epoch_val = {
            'loss_sum' : [],
            'loss_classifier' : [],
            'loss_box_reg' : [],
            'loss_objectness' : [],
            'loss_rpn_box_reg' : [] }
        running_loss_val = 0.0

        ################## TRAINING STARTS ######################## 
        model.train()
        for i, data in enumerate(train_loader):
            imgs, bboxes, labels = data
            # img = imgs[0].numpy().copy().transpose(1, 2, 0)
            # cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            images = list(im.to(device) for im in imgs)

            targets = [{'boxes': b.to(device), 'labels': l.to(device)} for b, l in zip(bboxes, labels)]
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            #apply weighting to losses
            if reg_weight != 1:
                loss_dict['loss_box_reg'] = loss_dict['loss_box_reg'] * reg_weight
            loss = sum(l for l in loss_dict.values())
            loss_value = loss.item()

            running_loss += loss_value 
            loss_per_epoch['loss_sum'].append(loss_value)
            loss_per_epoch['loss_classifier'].append(loss_dict['loss_classifier'].item())
            loss_per_epoch['loss_box_reg'].append(loss_dict['loss_box_reg'].item())
            loss_per_epoch['loss_objectness'].append(loss_dict['loss_objectness'].item())
            loss_per_epoch['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].item())

            if (i + 1) % log_interval == 0:
                print('%s ::Training:: [%d, %5d] loss: %.5f' %
                    (model_number, epoch + 1, i + 1, running_loss / log_interval))
                print([(k, v.item()) for k, v in loss_dict.items()])

                loss_total_dict['loss_sum'].append(sum(j for j in loss_per_epoch['loss_sum']) / i) 
                loss_total_dict['loss_classifier'].append(sum(j for j in loss_per_epoch['loss_classifier']) / i) 
                loss_total_dict['loss_box_reg'].append(sum(j for j in loss_per_epoch['loss_box_reg']) / i) 
                loss_total_dict['loss_objectness'].append(sum(j for j in loss_per_epoch['loss_objectness']) / i) 
                loss_total_dict['loss_rpn_box_reg'].append(sum(j for j in loss_per_epoch['loss_rpn_box_reg']) / i) 

                running_loss = 0.0
                if epoch in checkpoints: 
                    print(f"Saving net at: {model_save_path_new}")
                    torch.save(model.state_dict(), model_save_path_new + 'e' + f'{epoch}' + '.th')
                    
                with open(f'{model_save_path_new}-train', 'wb') as filezin:
                    pickle.dump(loss_total_dict, filezin)

            loss.backward()

            optimizer.step()

        ################## VALIDATION STARTS ######################## 
        for i, data in enumerate(val_loader):
            imgs, bboxes, labels = data
            #TODO: ASK PAULO
            img = imgs[0].numpy().copy().transpose(1, 2, 0)
            images = list(im.to(device) for im in imgs)
            targets = [{'boxes': b.to(device), 'labels': l.to(device)} for b, l in zip(bboxes, labels)]

            #running model
            loss_dict = model(images, targets)
            #apply weighting to losses
            if reg_weight != 1:
                loss_dict['loss_box_reg'] = loss_dict['loss_box_reg'] * reg_weight
            loss = sum(l for l in loss_dict.values())
            loss_value = loss.item()

            running_loss += loss_value 
            loss_per_epoch_val['loss_sum'].append(loss_value)
            loss_per_epoch_val['loss_classifier'].append(loss_dict['loss_classifier'].item())
            loss_per_epoch_val['loss_box_reg'].append(loss_dict['loss_box_reg'].item())
            loss_per_epoch_val['loss_objectness'].append(loss_dict['loss_objectness'].item())
            loss_per_epoch_val['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].item())

            # model.eval()

            # #forward prop
            # bboxes_pred, pred_cls, pred_scores = fastercnn.get_prediction_fastercnn(
            #     imgs[0].to(device), net, threshold, category_names=categories, img_is_path=False)
            # cls_gt = np.array(categories)[[t.item() for t in targets[0]]]

            if (i + 1) % log_interval_val == 0:
                print('%s ::Validation:: [%d, %5d] loss: %.5f' %
                    (model_number, epoch + 1, i + 1, running_loss / log_interval_val))
                print(f'Taking (precisely) {(time()-tic)/60} minutes per epoch')
                print([(k, v.item()) for k, v in loss_dict.items()])

                loss_total_dict_val['loss_sum'].append(sum(j for j in loss_per_epoch_val['loss_sum']) / i) 
                loss_total_dict_val['loss_classifier'].append(sum(j for j in loss_per_epoch_val['loss_classifier']) / i) 
                loss_total_dict_val['loss_box_reg'].append(sum(j for j in loss_per_epoch_val['loss_box_reg']) / i) 
                loss_total_dict_val['loss_objectness'].append(sum(j for j in loss_per_epoch_val['loss_objectness']) / i) 
                loss_total_dict_val['loss_rpn_box_reg'].append(sum(j for j in loss_per_epoch_val['loss_rpn_box_reg']) / i) 

                running_loss = 0.0

                if epoch in checkpoints: 
                    loss_total_dict['loss_sum_val'] = loss_total_dict_val['loss_sum']
                    loss_total_dict['loss_classifier_val'] = loss_total_dict_val['loss_sum']
                    loss_total_dict['loss_box_reg_val'] = loss_total_dict_val['loss_sum']
                    loss_total_dict['loss_objectiveness_val'] = loss_total_dict_val['loss_sum']
                    loss_total_dict['loss_rpn_box_reg_val'] = loss_total_dict_val['loss_sum']
                    with open(f'{model_save_path_new}-train', 'wb') as filezin:
                        pickle.dump(loss_total_dict, filezin)