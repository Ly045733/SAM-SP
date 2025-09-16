import torch

import numpy as np 
from tqdm import tqdm 
from typing import Tuple
from torchvision.transforms.functional import to_pil_image

from segment_anything.utils.transforms import ResizeLongestSide
from my_utils.Sampling_Combine import random_sample, contour_sample, combine, Entropy_Grids_Sampling,Entropy_contour_Sampling,contour_sample_without_bs,process_input_SAM,combine_cell,process_input_SAM_cell
# from .make_prompt import *
from .metrics import *
from my_utils.metrics import calculate_metrics
from matplotlib import pyplot as plt




def model_train(
    model,
    data_loader,
    criterion,
    optimizer,        
    device,
    scheduler,
    num,
    sigClassLoss
) -> Tuple[float, float, float, float]:
    """
    Train the model

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions 
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """
    
    # Training
    model.train()
    
    running_iouloss = 0.0
    running_diceloss = 0.0
    
    running_dice = 0.0
    running_iou = 0.0
    
    n_data = 0
    
    diceloss = criterion[0]    
    iouloss = criterion[1]
    celoss = criterion[2]
    pdloss = criterion[3]

    transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
    loss_all = 0.0
    for Sample_List in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels, PLs, scribbles, id = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["pesudo_label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
        labels_np = np.array(labels.cpu())
        
        # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
        
        batched_input_LV, batched_input_MYO, batched_input_RV = [], [], []
        for image, label,PL, scribble in zip(images,labels,PLs, scribbles):
            # prepare image
            original_size = image.shape[1:3]
            image_RGB = torch.cat([image, image, image], dim=0)

            image_RGB = transform.apply_image(image_RGB)
            image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
            image_RGB = image_RGB.permute(2, 0, 1).contiguous()
            
            # sampled_point_batch_LV =  Entropy_contour_Sampling(scribble, image, 1, num)
            # sampled_point_batch_MYO = Entropy_contour_Sampling(scribble, image, 2, num)
            # sampled_point_batch_RV =  Entropy_contour_Sampling(scribble, image, 3, num)
            # sampled_point_batch_background = Entropy_contour_Sampling(scribble, image, 4, num)

            sampled_point_batch_LV =  contour_sample_without_bs(scribble, 1, num)
            sampled_point_batch_MYO = contour_sample_without_bs(scribble, 2, num)
            sampled_point_batch_RV =  contour_sample_without_bs(scribble, 3, num)
            sampled_point_batch_background = contour_sample_without_bs(scribble, 4, num)

            all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO = combine(sampled_point_batch_LV, sampled_point_batch_RV, sampled_point_batch_MYO, sampled_point_batch_background)
            batched_input_LV, batched_input_MYO, batched_input_RV = process_input_SAM(transform,image_RGB, original_size, all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO, batched_input_LV, batched_input_MYO, batched_input_RV)
            
        batched_inputs = [batched_input_LV, batched_input_MYO, batched_input_RV]
        if  sigClassLoss==1:
            loss = 0
            for i, batched_input_cls in enumerate(batched_inputs):
                batched_output = model(batched_input_cls, multimask_output=False)
                pred_mask = torch.stack([m["masks"][0][0] for m in batched_output])
                iou_loss_ = iouloss(pred_mask.unsqueeze(1), (PLs==i+1).unsqueeze(1))
                dice_loss_ = diceloss(pred_mask, (PLs==i+1)) 
                # pCE_loss_ = celoss(pred_mask, (scribbles==i+1)[:].long())
                loss += iou_loss_ + dice_loss_
                loss_all += loss
                # loss.backward()
                # optimizer.step()
        else:
            batched_output_masks,batched_output_masks_pred = -1e5*torch.zeros((images.shape[0],4,images.shape[2],images.shape[3])).cuda(),-1e5*torch.zeros((images.shape[0],4,images.shape[2],images.shape[3])).cuda()
            masks = torch.zeros((images.shape[2],images.shape[3])).cuda()
            masks_pred = torch.zeros((images.shape[2],images.shape[3])).cuda()
            for i, batched_input_cls in enumerate(batched_inputs):
                batched_output = model(batched_input_cls, multimask_output=False)
                for j in range(data_loader.batch_size):
                    batched_output_masks[j,i+1,:,:] = batched_output[j]["masks"][0][0]
                    # batched_output_masks_pred[j,i+1,:,:] = batched_output[j]["masks_pred"][0][0]
                    masks += batched_output_masks[j,i+1,:,:]
                masks = masks/3
                batched_output_masks[0,i+1,:,:] = masks


            iou_loss_ = iouloss(batched_output_masks, PLs.unsqueeze(1))
            dice_loss_ = pdloss(batched_output_masks, PLs.unsqueeze(1)) 
            pCE_loss_ = celoss(batched_output_masks, scribbles[:].long())
            loss = iou_loss_ + dice_loss_ + pCE_loss_
            loss_all += loss
        loss.backward()
        optimizer.step()

    



    if scheduler:
        scheduler.step()
    avg_loss_all = loss_all / len(data_loader)

    return avg_loss_all


def model_evaluate(
    model,
    data_loader,
    criterion,
    device,
    num 
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model

    Args:
        model (nn.Module): SAM model 
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """

    # Evaluation
    model.eval()
    total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    # len_data = len(data_loader.len())
    with torch.no_grad():
        
        running_iouloss = 0.0
        running_diceloss = 0.0
        
        running_dice = 0.0
        running_iou = 0.0
        
        diceloss = criterion[0]        
        iouloss = criterion[1]
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        dice_all = 0.0
        iou_all = 0.0
        len = 0
        for Sample_List in tqdm(data_loader): 
            images, labels, scribbles = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["scribble"].cuda()
            images, labels, scribbles = images.permute(1,0,2,3), labels.permute(1,0,2,3), scribbles.permute(1,0,2,3)
            # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            
            for bs in range(images.shape[0]):
                batched_input_LV, batched_input_MYO, batched_input_RV = [], [], []
                image_bs,label_bs,scribble_bs = images[bs],labels[bs],scribbles[bs]
                for image, label,scribble in zip(image_bs,label_bs, scribble_bs):
                    # prepare image
                    original_size = image.shape[0:3]
                    image = image.unsqueeze(0)
                    image_RGB = torch.cat([image, image, image], dim=0)

                    image_RGB = transform.apply_image(image_RGB)
                    image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
                    image_RGB = image_RGB.permute(2, 0, 1).contiguous()
                    
                    # sampled_point_batch_LV =  max_distance_sample(label.unsqueeze(0),scribble.unsqueeze(0), 1, num)
                    # sampled_point_batch_MYO = max_distance_sample(label.unsqueeze(0),scribble.unsqueeze(0), 2, num)
                    # sampled_point_batch_RV =  max_distance_sample(label.unsqueeze(0),scribble.unsqueeze(0), 3, num)
                    # sampled_point_batch_background = contour_sample_without_bs(scribble, 4, num)
                    
                    sampled_point_batch_LV =  contour_sample_without_bs(scribble, 1, num)
                    sampled_point_batch_MYO = contour_sample_without_bs(scribble, 2, num)
                    sampled_point_batch_RV =  contour_sample_without_bs(scribble, 3, num)
                    sampled_point_batch_background = contour_sample_without_bs(scribble, 4, num)

                    all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO = combine(sampled_point_batch_LV, sampled_point_batch_RV, sampled_point_batch_MYO, sampled_point_batch_background)
                    batched_input_LV, batched_input_MYO, batched_input_RV = process_input_SAM(transform,image_RGB, original_size, all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO, batched_input_LV, batched_input_MYO, batched_input_RV)
                    # plt.figure(figsize=(10,10))
                    # plt.imshow(image[0].cpu().numpy())
                    # plt.imshow(label.cpu().numpy(),alpha=0.5,cmap='gray')
                    # if sampled_point_batch_LV is not None:
                    #     for i in range(sampled_point_batch_LV.shape[0]):
                    #         x, y  = sampled_point_batch_LV[i]
                    #         plt.plot(x, y, 'ro')
                    # plt.axis('off')
                    # plt.savefig(f'data/222/{111}_mask_LV_.png', bbox_inches='tight', pad_inches=0) 
                    # plt.close()

                    # plt.figure(figsize=(10,10))
                    # plt.imshow(image[0].cpu().numpy())
                    # plt.imshow(label.cpu().numpy(),alpha=0.5,cmap='gray')
                    # if sampled_point_batch_RV is not None:
                    #     for i in range(sampled_point_batch_RV.shape[0]):
                    #         x, y  = sampled_point_batch_RV[i]
                    #         plt.plot(x, y, 'ro')

                    # plt.axis('off')
                    # plt.savefig(f'data/222/{111}_mask_RV_.png', bbox_inches='tight', pad_inches=0) 
                    # plt.close()

                    # plt.figure(figsize=(10,10))
                    # plt.imshow(image[0].cpu().numpy())
                    # plt.imshow(label.cpu().numpy(),alpha=0.5,cmap='gray')
                    # if sampled_point_batch_MYO is not None:
                    #     for i in range(sampled_point_batch_MYO.shape[0]):
                    #         x, y  = sampled_point_batch_MYO[i]
                    #         plt.plot(x, y, 'ro')
                    # plt.axis('off')
                    # plt.savefig(f'data/222/{111}_mask_MYO_.png', bbox_inches='tight', pad_inches=0) 
                    # plt.close()

                batched_output_LV = model(batched_input_LV, multimask_output=False)
                batched_output_MYO = model(batched_input_MYO, multimask_output=False)
                batched_output_RV = model(batched_input_RV, multimask_output=False)

                masks_LV =   batched_output_LV[0]['masks_pred'][0][0].cpu().numpy()
                masks_MYO =  batched_output_MYO[0]['masks_pred'][0][0].cpu().numpy()
                masks_RV =   batched_output_RV[0]['masks_pred'][0][0].cpu().numpy()
                mask = np.zeros((masks_LV.shape[0], masks_LV.shape[1]), np.uint8)
                if all_points_MYO is not None:
                    mask[masks_MYO != False] = 2
                if all_points_LV is not None:
                    mask[masks_LV != False]  = 1
                if all_points_RV is not None:
                    mask[masks_RV != False]  = 3
                if all_points_MYO is None and all_points_LV is not None and all_points_RV is not None:
                    mask = np.zeros((masks_LV.shape[0], masks_LV.shape[1]), np.uint8)
                mask = torch.tensor(np.expand_dims(mask, axis = 0)).cuda()
                # plt.figure(figsize=(10,10))
                # plt.imshow(image[0].cpu().numpy())
                # plt.imshow(masks_MYO,alpha=0.5,cmap='gray')
                # if sampled_point_batch_MYO is not None:
                #     for i in range(sampled_point_batch_MYO.shape[0]):
                #         x, y  = sampled_point_batch_MYO[i]
                #         plt.plot(x, y, 'ro')
                # plt.axis('off')
                # plt.savefig(f'data/222/{222}_mask_MYO_.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(image[0].cpu().numpy())
                # plt.imshow(masks_LV,alpha=0.5,cmap='gray')
                # if sampled_point_batch_LV is not None:
                #     for i in range(sampled_point_batch_LV.shape[0]):
                #         x, y  = sampled_point_batch_LV[i]
                #         plt.plot(x, y, 'ro')
                # plt.axis('off')
                # plt.savefig(f'data/222/{222}_mask_LV_.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(image[0].cpu().numpy())
                # plt.imshow(masks_RV,alpha=0.5,cmap='gray')
                # if sampled_point_batch_RV is not None:
                #     for i in range(sampled_point_batch_RV.shape[0]):
                #         x, y  = sampled_point_batch_RV[i]
                #         plt.plot(x, y, 'ro')
                # plt.axis('off')
                # plt.savefig(f'data/222/{222}_mask_RV_.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()


                for j, gt_mask in enumerate(label_bs):
                    
                    if np.unique(gt_mask.cpu().numpy()).any() != 0:
                        gt_mask = gt_mask.unsqueeze(0)

                        dices, HD95,IOU = calculate_metrics(gt_mask, mask, 4)
                        
                            
                        for cls in range(3):
                            total_dice_scores[cls] += dices[cls]
                            total_HD95_scores[cls] += HD95[cls]
                            total_IOU_scores[cls] += IOU[cls]
                        
                    else:
                    ### loss & metrics ###
                        for cls in range(3):
                            total_dice_scores[cls] += 1
                            total_HD95_scores[cls] += 0
                            total_IOU_scores[cls] += 1
                           
                ### update loss & metrics ###

                del image_bs,label_bs,scribble_bs,batched_input_LV, batched_input_MYO, batched_input_RV  # 显式删除无用变量
                torch.cuda.empty_cache()

            len += images.shape[0]

        
        ### Average loss & metrics ### 
        
        avg_dice_scores = [(total / len).cpu().numpy() for total in total_dice_scores]
        avg_HD95_scores = [(total / len) for total in total_HD95_scores]
        avg_IOU_scores =  [(total / len).cpu().numpy() for total in total_IOU_scores]
        
        avg_dice = np.mean(avg_dice_scores)
        avg_HD95 = np.mean(avg_HD95_scores)
        avg_IOU =  np.mean(avg_IOU_scores)


    return avg_dice, avg_HD95,avg_IOU

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', verbose=True):
        """
        Pytorch Early Stopping

        Args:
            patience (int, optional): patience. Defaults to 10.
            delta (float, optional): threshold to update best score. Defaults to 0.0.
            mode (str, optional): 'min' or 'max'. Defaults to 'min'(comparing loss -> lower is better).
            verbose (bool, optional): verbose. Defaults to True.
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        
    def __call__(self, score):
        # self.best_score = 15.0
        # _1 = np.abs(self.best_score - score.cpu())
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}')
                    # , Delta: {np.abs(self.best_score - score.cpu().numpy()):.5f}
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}')
                
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False

def model_train_cell(
    model,
    data_loader,
    criterion,
    optimizer,        
    device,
    scheduler,
    num
) -> Tuple[float, float, float, float]:
    """
    Train the model

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions 
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """
    
    # Training
    model.train()
    
    running_iouloss = 0.0
    running_diceloss = 0.0
    
    running_dice = 0.0
    running_iou = 0.0
    
    n_data = 0
    
    diceloss = criterion[0]    
    iouloss = criterion[1]
   
    transform = ResizeLongestSide(target_length=model.image_encoder.img_size)

    for Sample_List in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels, PLs, scribbles, id = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["pesudo_label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
        labels_np = np.array(labels.cpu())
        
        # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
        
        batched_input_Nu, batched_input_Cy = [], []
        for image, scribble in zip(images, scribbles):
            # prepare image
            original_size = image.shape[1:3]
            # image_RGB = torch.cat([image, image, image], dim=0)
            # print(image.shape)    
            image_RGB = transform.apply_image(image)
            # print(image_RGB.shape)  
            image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
            image_RGB = image_RGB.permute(2, 0, 1).contiguous()
            
            # sampled_point_batch_LV =  Entropy_contour_Sampling(scribble, image, 1, num)
            # sampled_point_batch_MYO = Entropy_contour_Sampling(scribble, image, 2, num)
            # sampled_point_batch_RV =  Entropy_contour_Sampling(scribble, image, 3, num)
            # sampled_point_batch_background = Entropy_contour_Sampling(scribble, image, 4, num)

            sampled_point_batch_Nu =  contour_sample_without_bs(scribble, 1, num)
            sampled_point_batch_Cy = contour_sample_without_bs(scribble, 2, num)
            # sampled_point_batch_RV =  contour_sample(scribble, 3, num)
            sampled_point_batch_background = contour_sample_without_bs(scribble, 0, num)
            all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy = combine_cell(sampled_point_batch_Nu, sampled_point_batch_Cy, sampled_point_batch_background)
            batched_input_Nu, batched_input_Cy = process_input_SAM_cell(transform,image_RGB, original_size,all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy, batched_input_Nu, batched_input_Cy)
            
            
        batched_inputs = [batched_input_Nu, batched_input_Cy]
        for i, batched_input_cls in enumerate(batched_inputs):

            batched_output = model(batched_input_cls, multimask_output=False)
        
            loss = 0.0
            iou_loss = 0.0
            dice_loss = 0.0

            dice = 0.0
            iou = 0.0
            PL_masks = PLs == i+1
            for j, pl_mask in enumerate(PL_masks):
                
                masks = batched_output[j]['masks']
                masks_pred = batched_output[j]['masks_pred']

            #     plt.figure(figsize=(10,10))
            #     plt.imshow(images[j].permute(1,2,0).cpu().numpy())
            #     plt.axis('off')
            # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
            #     plt.imshow(masks_pred.squeeze(0).squeeze(0).cpu().numpy(),alpha=0.5,cmap='gray')
            #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
            #     plt.savefig(f'data/222/{id[j]}_mask_{i}_.png', bbox_inches='tight', pad_inches=0) 
            #     plt.close()

            #     plt.figure(figsize=(10,10))
            #     plt.imshow(images[j].permute(1,2,0).cpu().numpy())
            #     plt.axis('off')
            # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
            #     plt.imshow(pl_mask.cpu().numpy(),alpha=0.5,cmap='gray')
            #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
            #     plt.savefig(f'data/222/{id[j]}_GT_{i}_.png', bbox_inches='tight', pad_inches=0) 
            #     plt.close()
                
                
                ## loss & metrics ###
                iou_loss_ = iouloss(masks.squeeze(1), pl_mask.unsqueeze(0))
                dice_loss_ = diceloss(masks.squeeze(1), pl_mask.unsqueeze(0)) 
                loss_ = dice_loss_ + iou_loss_
                
                dice_ = Dice(masks_pred.squeeze(1), pl_mask.unsqueeze(0))
                iou_ = IoU(masks_pred.squeeze(1), pl_mask.unsqueeze(0))
                            
                loss = loss + loss_
                iou_loss = iou_loss + iou_loss_
                dice_loss = dice_loss + dice_loss_

                dice = dice + dice_
                iou = iou + iou_
            
            # average loss & metrcis (mini-batch)
            loss = loss / labels.shape[0]
            
            iou_loss = iou_loss / labels.shape[0]
            dice_loss = dice_loss / labels.shape[0]

            dice = dice / labels.shape[0]
            iou = iou / labels.shape[0]
                    
            loss.backward()
            optimizer.step()
                
            ### update loss & metrics ###
            
            running_iouloss += iou_loss.item() * images.size(0)
            running_diceloss += dice_loss.item() * images.size(0)

            running_dice += dice.item() * images.size(0)
            running_iou += iou.item() * images.size(0)
            
            n_data += images.size(0)
        
    if scheduler:
        scheduler.step()
    n_data = n_data / 3
    ### Average loss & metrics ###
    avg_iou_loss = running_iouloss / n_data
    avg_dice_loss = running_diceloss / n_data

    avg_dice = running_dice / n_data
    avg_iou = running_iou / n_data 

    return avg_dice_loss, avg_iou_loss, avg_dice, avg_iou


def model_evaluate_cell(
    model,
    data_loader,
    criterion,
    device,
    num 
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model

    Args:
        model (nn.Module): SAM model 
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """

    # Evaluation
    model.eval()
    total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    len_data = len(data_loader.dataset)
    with torch.no_grad():
        
        running_iouloss = 0.0
        running_diceloss = 0.0
        
        running_dice = 0.0
        running_iou = 0.0
        
        diceloss = criterion[0]        
        iouloss = criterion[1]
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        
        for Sample_List in tqdm(data_loader): 
            images, labels, scribbles,id  = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
            # images, labels, scribbles = images.permute(1,0,2,3), labels.permute(1,0,2,3), scribbles.permute(1,0,2,3)
            # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            batched_input_Nu, batched_input_Cy = [], []
            # image_bs,label_bs,scribble_bs = images[bs],labels[bs],scribbles[bs]
            for image, scribble in zip(images, scribbles):
                # prepare image
                original_size = image.shape[1:3]
                # image = image.unsqueeze(0)
                # image_RGB = torch.cat([image, image, image], dim=0)

                # print(image.shape)    
                image_RGB = transform.apply_image(image)
                # print(image_RGB.shape)  
                image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
                image_RGB = image_RGB.permute(2, 0, 1).contiguous()
                
                # sampled_point_batch_LV =  Entropy_contour_Sampling(scribble, image, 1, num)
                # sampled_point_batch_MYO = Entropy_contour_Sampling(scribble, image, 2, num)
                # sampled_point_batch_RV =  Entropy_contour_Sampling(scribble, image, 3, num)
                # sampled_point_batch_background = Entropy_contour_Sampling(scribble, image, 4, num)

                sampled_point_batch_Nu =  contour_sample_without_bs(scribble, 1, num)
                sampled_point_batch_Cy = contour_sample_without_bs(scribble, 2, num)
                # sampled_point_batch_RV =  contour_sample(scribble, 3, num)
                sampled_point_batch_background = contour_sample_without_bs(scribble, 0, num)
                all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy = combine_cell(sampled_point_batch_Nu, sampled_point_batch_Cy, sampled_point_batch_background)
                batched_input_Nu, batched_input_Cy = process_input_SAM_cell(transform,image_RGB, original_size,all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy, batched_input_Nu, batched_input_Cy)
        
            
            
            
                        
                # batched_inputs = [batched_input_LV, batched_input_MYO, batched_input_RV]

            batched_output_Nu = model(batched_input_Nu, multimask_output=False)
            batched_output_Cy = model(batched_input_Cy, multimask_output=False)
            # batched_output_RV = model(batched_input_RV, multimask_output=False)

            masks_Nu =   batched_output_Nu[0]['masks_pred'][0][0].cpu().numpy()
            masks_Cy =  batched_output_Cy[0]['masks_pred'][0][0].cpu().numpy()
            # masks_RV =   batched_output_RV[0]['masks_pred'][0][0].cpu().numpy()

            mask = np.zeros((masks_Nu.shape[0], masks_Nu.shape[1]), np.uint8)
            if all_points_Cy is not None:
                mask[masks_Cy != False] = 2
            if all_points_Nu is not None:
                mask[masks_Nu != False]  = 1
                


            mask = torch.tensor(np.expand_dims(mask, axis = 0)).cuda()
            loss = 0.0
            iou_loss = 0.0
            dice_loss = 0.0

            dice = 0.0
            iou = 0.0
            
            for j, gt_mask in enumerate(labels):
                # print(np.unique(gt_mask.cpu().numpy()))

            #     plt.figure(figsize=(10,10))
            #     plt.imshow(images[j].permute(1,2,0).cpu().numpy())
            #     plt.axis('off')
            # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
            #     plt.imshow(mask.squeeze(0).squeeze(0).cpu().numpy(),alpha=0.5)
            #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
            #     plt.savefig(f'data/222/{id[j]}_mask_.png', bbox_inches='tight', pad_inches=0) 
            #     plt.close()

            #     plt.figure(figsize=(10,10))
            #     plt.imshow(images[j].permute(1,2,0).cpu().numpy())
            #     plt.axis('off')
            # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
            #     plt.imshow(gt_mask.cpu().numpy(),alpha=0.5)
            #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
            #     plt.savefig(f'data/222/{id[j]}_GT_.png', bbox_inches='tight', pad_inches=0) 
            #     plt.close()
                if np.unique(gt_mask.cpu().numpy()).any() != 0:
                    gt_mask = gt_mask.unsqueeze(0)
                    iou_loss1 = iouloss((mask==1),(gt_mask==1))
                    dice_loss1 = diceloss((mask==1), (gt_mask==1))

                    iou_loss2 = iouloss(mask==2, gt_mask==2 )
                    dice_loss2 = diceloss(mask==2, gt_mask==2)


                    dices, HD95,IOU = calculate_metrics(gt_mask, mask, 3)
                    for cls in range(2):
                        total_dice_scores[cls] += dices[cls]/labels.shape[0]
                        total_HD95_scores[cls] += HD95[cls]/labels.shape[0]
                        total_IOU_scores[cls] += IOU[cls]/labels.shape[0]
                    
                    iou_loss = iou_loss + iou_loss1 + iou_loss2 
                    dice_loss = dice_loss + dice_loss1 + dice_loss2 

                    
                else:
                ### loss & metrics ###
                    iou_loss = iou_loss + 0.0
                    dice_loss = dice_loss + 0.0
                    dice = dice + 1.0
                    iou = iou + 1.0
                
            iou_loss = iou_loss / labels.shape[0]
            dice_loss = dice_loss / labels.shape[0]
            
            ### update loss & metrics ###

            running_iouloss += iou_loss * images.shape[0]
            running_diceloss += dice_loss * images.shape[0]


                # del image_bs,label_bs,scribble_bs,batched_input_LV, batched_input_MYO, batched_input_RV  # 显式删除无用变量
                # torch.cuda.empty_cache()
        
        ### Average loss & metrics ### 
        
        avg_dice_scores = [(total / len_data).cpu().numpy() for total in total_dice_scores]
        avg_HD95_scores = [(total / len_data) for total in total_HD95_scores]
        avg_IOU_scores =  [(total / len_data).cpu().numpy() for total in total_IOU_scores]
        
        avg_dice = np.mean(avg_dice_scores)
        avg_HD95 = np.mean(avg_HD95_scores)
        avg_IOU =  np.mean(avg_IOU_scores)

        avg_iou_loss = running_iouloss / len_data
        avg_dice_loss = running_diceloss / len_data
        # avg_dice = running_dice / len_data
        # avg_iou = running_iou / len_data  
        print(f'category LV:   Dice: {avg_dice_scores[0]:.3f},  HD95: {avg_HD95_scores[0]:.3f},  IOU: {avg_IOU_scores[0]:.3f}')
        print(f'category MYO:  Dice: {avg_dice_scores[1]:.3f},  HD95: {avg_HD95_scores[1]:.3f},  IOU: {avg_IOU_scores[1]:.3f}')
        print(f'Total:       Dice: {avg_dice:.3f},  HD95: {avg_HD95:.3f},  IoU: {avg_IOU:.3f}')

    return avg_dice_loss/3, avg_iou_loss/3, avg_dice, avg_HD95,avg_IOU
