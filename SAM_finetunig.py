import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import logging
import os
import argparse
import numpy as np 
from losses import SAM_loss
from segment_anything import sam_model_registry
from my_utils import SAM_trainer,save_weight
# from dataset.dataset_MSCMR import BaseDataSets_SAM,RandomGenerator_SAM
from dataset.dataset_ACDC import BaseDataSets_SAM_Fintune,RandomGenerator_SAM_Finttune
from torchvision import transforms
import random
from torch.nn.modules.loss import CrossEntropyLoss
from PIL import Image
from my_utils.metrics import calculate_metrics
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from my_utils.Sampling_Combine import random_sample, contour_sample, combine, Entropy_Grids_Sampling,Entropy_contour_Sampling,contour_sample_without_bs,process_input_SAM

# from tools import seed, dataset, losses, save_weight, generate_sam_mask

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root_path', type=str, default='/home/lxy/pycharm_project/model_New/Dataset/ACDC')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size allocated to each GPU')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--num', type=int, default=10)
    
    parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default='/home/lxy/pycharm_project/model_New/SAM_model_path/sam_vit_h_4b8939.pth', help='SAM model checkpoint')
    parser.add_argument('--exp', type=str, default='SAM_FineTune/iteration1_vith')
    parser.add_argument('--iter', type=str, default='iteration1')
    
    parser.add_argument('--max_epochs', type=int, default=20, help='total epoch')
    parser.add_argument('--base_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--deterministic', type=int,  default=1,help='whether use deterministic training')
    parser.add_argument('--ratio', type=float, default=0.05, help='weight decay')
    parser.add_argument('--sigClassLoss', type=int, default=0, help='Loss Computer Method')
    return parser

def main(opts, snapshot_path, savepath):
    
    num_classes = opts.num_classes
    max_iterations = opts.max_epochs
    sigClassLoss = opts.sigClassLoss
    ### Dataset & Dataloader ### 
    train_set = BaseDataSets_SAM_Fintune(base_dir=opts.root_path, fold='fold1',transform=transforms.Compose(
        [RandomGenerator_SAM_Finttune(opts.patch_size,split='train')]), split="train",pesudo_label = 'iter0',ratio=opts.ratio)
    

    val_set = BaseDataSets_SAM_Fintune(base_dir=opts.root_path, fold='fold1', split="val", transform=transforms.Compose(
        [RandomGenerator_SAM_Finttune(opts.patch_size,split='val')]),pesudo_label = 'iter0',ratio=0.1)

    train_loader = DataLoader(
        train_set, 
        batch_size=opts.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        shuffle=False, drop_last=True
    )
    
    ### Model config ### 
    
    sam_checkpoint = opts.sam_checkpoint
    model_type = opts.sam_model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.cuda()

    # set trainable parameters
    for _, p in sam.image_encoder.named_parameters():
        p.requires_grad = False
        
    for _, p in sam.prompt_encoder.named_parameters():
        p.requires_grad = False

    # fine-tuning mask decoder         
    for _, p in sam.mask_decoder.named_parameters():
        p.requires_grad = True
        
        
    ### Training config ###  
   
    iouloss = SAM_loss.IoULoss()
    diceloss = SAM_loss.DiceLoss()
    PDDice = SAM_loss.pDLoss(4,ignore_index=4)
    ce_loss = CrossEntropyLoss(ignore_index=4)

    es = SAM_trainer.EarlyStopping(patience=10, delta=0, mode='min', verbose=True)
    optimizer = torch.optim.AdamW(sam.parameters(), lr=opts.base_lr, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )
    
    max_loss = np.inf
    best_dice = 0.0
    ### Training phase ###
    
    for epoch in range(opts.max_epochs):
        print(f'# Epochs {epoch}')
        train_dice_loss = SAM_trainer.model_train(
            model=sam,
            data_loader=train_loader,
            criterion=[diceloss, iouloss,ce_loss,PDDice],
            optimizer=optimizer,
            device='cuda',
            scheduler=scheduler,
            num = opts.num,
            sigClassLoss = sigClassLoss
        )

        val_dice, val_HD95, val_iou = SAM_trainer.model_evaluate(
            model=sam,
            data_loader=val_loader,
            criterion=[diceloss, iouloss],
            device='cuda',
            num = opts.num
        )
        
        # val_loss = train_dice_loss + train_iou_loss
        
        # check EarlyStopping
        # if val_loss.device.type=='cuda:0':
        #     val_loss = val_loss.cpu()
        # es(val_dice)
        save_best_path = os.path.join(savepath,'sam_best_decoder.pth')
        save_best_path_all_perepoch = os.path.join(savepath,f'sam_all_{epoch}_{val_dice}.pth')
        save_best_path_all = os.path.join(savepath,f'sam_best_all.pth')
        # save best model 
        if val_dice > best_dice:
            print(f'[INFO] val_dice has been improved from {best_dice:.5f} to {val_dice:.5f}. Save model.')
            best_dice = val_dice
            _ = save_weight.save_partial_weight(model=sam, save_path=save_best_path)
            torch.save(sam.state_dict(), save_best_path_all)
            torch.save(sam.state_dict(), save_best_path_all_perepoch)
            print("save model to {}".format(save_best_path_all))
        if epoch%3 ==0:
            torch.save(sam.state_dict(), save_best_path_all_perepoch)
        # print current loss & metric
        print(f'epoch {epoch+1:02d}, dice_loss: {train_dice_loss:.5f} \n')
        print(f'val_dice: {val_dice:.5f}, val_iou: {val_iou:.5f}, val_HD95:{val_HD95:.5f} \n')
        
        if es.early_stop:
            break   
    # len_data = len(val_loader.dataset)
    # sam.eval()
    # result_path = "/home/cj/code/SAM_Scribble/result" + "/pesudo_label_image"
    # os.makedirs(result_path, exist_ok = True)
    # running_iouloss = 0.0
    # running_diceloss = 0.0
    
    # running_dice = 0.0
    # running_iou = 0.0
    # test_set = BaseDataSets111(base_dir=opts.root_path, fold='fold1',transform=transforms.Compose([
    #                             RandomGenerator111([256, 256])]
    #                             ), split="train",pesudo_label = 'SAM_iteration1')
    # test_loader = DataLoader(
    #     train_set, 
    #     batch_size=1, 
    #     shuffle=False, 
    # )
    # with torch.no_grad():
    #     transform = ResizeLongestSide(target_length=sam.image_encoder.img_size)
    #     total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]    
    #     for sample_list in tqdm(test_loader): 
    #         images, labels, scribble, id  = sample_list['image'].cuda(), sample_list['label'].cuda(), sample_list['scribble'].cuda(), sample_list['idx']
    #         # images, labels, scribbles = images.permute(1,0,2,3), labels.permute(1,0,2,3), scribbles.permute(1,0,2,3)
    #         # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
    #         image = (np.array(image[0].permute(1,2,0).cpu())*255).astype(np.uint8)
    #         label = label[0].cpu().numpy()
    #         for bs in range(images.shape[0]):
    #             batched_input_LV, batched_input_MYO, batched_input_RV = [], [], []
    #             image_bs,label_bs,scribble_bs = images[bs],labels[bs],scribbles[bs]
    #             for image, scribble in zip(image_bs, scribble_bs):
    #                 # prepare image
    #                 original_size = image.shape[0:3]
    #                 image = image.unsqueeze(0)
    #                 image_RGB = torch.cat([image, image, image], dim=0)

    #                 image_RGB = transform.apply_image(image_RGB)
    #                 image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device='cuda')
    #                 image_RGB = image_RGB.permute(2, 0, 1).contiguous()
                    
    #                 # sampled_point_batch_LV =  Entropy_contour_Sampling(scribble, image, 1, num)
    #                 # sampled_point_batch_MYO = Entropy_contour_Sampling(scribble, image, 2, num)
    #                 # sampled_point_batch_RV =  Entropy_contour_Sampling(scribble, image, 3, num)
    #                 # sampled_point_batch_background = Entropy_contour_Sampling(scribble, image, 4, num)

    #                 sampled_point_batch_LV =  contour_sample_without_bs(scribble, 1, 5)
    #                 sampled_point_batch_MYO = contour_sample_without_bs(scribble, 2, 5)
    #                 sampled_point_batch_RV =  contour_sample_without_bs(scribble, 3, 5)
    #                 sampled_point_batch_background = contour_sample_without_bs(scribble, 4, 5)

    #                 all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO = combine(sampled_point_batch_LV, sampled_point_batch_RV, sampled_point_batch_MYO, sampled_point_batch_background)
    #                 batched_input_LV, batched_input_MYO, batched_input_RV = process_input_SAM(transform,image_RGB, original_size, all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO, batched_input_LV, batched_input_MYO, batched_input_RV)
            
                
                
                    
                        
    #             # batched_inputs = [batched_input_LV, batched_input_MYO, batched_input_RV]

    #             batched_output_LV = sam(batched_input_LV, multimask_output=False)
    #             batched_output_MYO = sam(batched_input_MYO, multimask_output=False)
    #             batched_output_RV = sam(batched_input_RV, multimask_output=False)

    #             masks_LV =   batched_output_LV[0]['masks_pred'][0][0].cpu().numpy()
    #             masks_MYO =  batched_output_MYO[0]['masks_pred'][0][0].cpu().numpy()
    #             masks_RV =   batched_output_RV[0]['masks_pred'][0][0].cpu().numpy()

    #             mask = np.zeros((masks_LV.shape[0], masks_LV.shape[1]), np.uint8)
    #             if all_points_LV is not None:
    #                 mask[masks_LV != False]  = 1
    #             if all_points_MYO is not None:
    #                 mask[masks_MYO != False] = 2
    #             if all_points_RV is not None:
    #                 mask[masks_RV != False]  = 3
    #             if all_points_MYO is None :
    #                 mask = np.zeros((masks_LV.shape[0], masks_LV.shape[0]), np.uint8)
        
    #             im = Image.fromarray(mask)
    #             print(mask.shape)
    #             im = im.convert('L')
    #             print(result_path)
    #             print(id)
    #             print(id.shape)
    #             im.save(f'{result_path}/{id[bs][0][:-3]}.png')
                
    #             mask = torch.tensor(np.expand_dims(mask, axis = 0)).cuda()
    #             loss = 0.0
    #             iou_loss = 0.0
    #             dice_loss = 0.0

    #             dice = 0.0
    #             iou = 0.0
                
    #             for j, gt_mask in enumerate(label_bs):
    #                 # print(np.unique(gt_mask.cpu().numpy()))
    #                 if np.unique(gt_mask.cpu().numpy()).any() != 0:
    #                     gt_mask = gt_mask.unsqueeze(0)
    #                     iou_loss1 = iouloss((mask==1),(gt_mask==1))
    #                     dice_loss1 = diceloss((mask==1), (gt_mask==1))

    #                     iou_loss2 = iouloss(mask==2, gt_mask==2 )
    #                     dice_loss2 = diceloss(mask==2, gt_mask==2)

    #                     iou_loss3 = iouloss(mask==3, gt_mask==3 )
    #                     dice_loss3 = diceloss(mask==3, gt_mask==3)

    #                     dices, HD95,IOU = calculate_metrics(gt_mask, mask, 4)
    #                     for cls in range(3):
    #                         total_dice_scores[cls] += dices[cls]/labels.shape[0]
    #                         total_HD95_scores[cls] += HD95[cls]/labels.shape[0]
    #                         total_IOU_scores[cls] += IOU[cls]/labels.shape[0]
                        
    #                     iou_loss = iou_loss + iou_loss1 + iou_loss2 + iou_loss3
    #                     dice_loss = dice_loss + dice_loss1 + dice_loss2 + dice_loss3

                        
    #                 else:
    #                 ### loss & metrics ###
    #                     iou_loss = iou_loss + 0.0
    #                     dice_loss = dice_loss + 0.0
    #                     dice = dice + 1.0
    #                     iou = iou + 1.0
                    
    #             iou_loss = iou_loss / labels.shape[0]
    #             dice_loss = dice_loss / labels.shape[0]
                
    #             ### update loss & metrics ###

    #             running_iouloss += iou_loss * images.shape[0]
    #             running_diceloss += dice_loss * images.shape[0]


    #             # del image_bs,label_bs,scribble_bs,batched_input_LV, batched_input_MYO, batched_input_RV  # 显式删除无用变量
    #             # torch.cuda.empty_cache()
        
    #     ### Average loss & metrics ### 
        
    #     avg_dice_scores = [(total / len_data).cpu().numpy() for total in total_dice_scores]
    #     avg_HD95_scores = [(total / len_data) for total in total_HD95_scores]
    #     avg_IOU_scores =  [(total / len_data).cpu().numpy() for total in total_IOU_scores]
        
    #     avg_dice = np.mean(avg_dice_scores)
    #     avg_HD95 = np.mean(avg_HD95_scores)
    #     avg_IOU =  np.mean(avg_IOU_scores)

    #     avg_iou_loss = running_iouloss / len_data
    #     avg_dice_loss = running_diceloss / len_data
    #     # avg_dice = running_dice / len_data
    #     # avg_iou = running_iou / len_data  
    #     print(f'category LV:   Dice: {avg_dice_scores[0]:.3f},  HD95: {avg_HD95_scores[0]:.3f},  IOU: {avg_IOU_scores[0]:.3f}')
    #     print(f'category MYO:  Dice: {avg_dice_scores[1]:.3f},  HD95: {avg_HD95_scores[1]:.3f},  IOU: {avg_IOU_scores[1]:.3f}')
    #     print(f'category RV:   Dice: {avg_dice_scores[2]:.3f},  HD95: {avg_HD95_scores[2]:.3f},  IOU: {avg_IOU_scores[2]:.3f}')
    #     print(f'Total:       Dice: {avg_dice:.3f},  HD95: {avg_HD95:.3f},  IoU: {avg_IOU:.3f}')


    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Iterative Re-Training', parents=[get_args_parser()])
    opts = parser.parse_args()

    if not opts.deterministic:
        opts.benchmark = True
        opts.deterministic = False
    else:
        opts.benchmark = False
        opts.deterministic = True
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)\
    
    savepath =  "../model_New/{}".format(
        opts.exp)
    snapshot_path = "../model_New/{}".format(
        opts.exp,)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # model_type =os.path.split(opts.checkpoint)[1][4:9]
    # sam_model = sam_model_registry[model_type](opts.checkpoint).cuda()
    # predictor = SamPredictor(sam_model)
    # dataset = BaseDataSets_SAM(base_dir="/home/lxy/pycharm_project/SAM_Scribble/data/ACDC", transform=transforms.Compose([
    #                             RandomGenerator_SAM([256, 256])]
    #                             ),split="train",  fold='fold1', sup_type='scribble')
    # dataloader = DataLoader(dataset, 1 , shuffle= False)
    logging.info(str(opts))

    print('=== Iterative Re-Training ===')
    
    print(f'# Iteration {opts.iter}')
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    main(opts, snapshot_path, savepath)
    
    print('=== DONE === \n')    