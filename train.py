import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from utils.utils import ORGAN_NAME, init_seed
from monai.inferers import sliding_window_inference
from model.SIPL_Model import SIPL
from dataset.dataloader import get_loader
from utils import loss
from utils.loss import get_aux_ce_loss, get_aux_dice_loss
from utils.utils import dice_score, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    loss_aux_ce_ave = 0
    loss_aux_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        torch.cuda.empty_cache()
        pred, class_wise_logits = model(x, is_train=True)
        # for i,l in enumerate(class_wise_logits):
        #     print('{}:{}'.format(i,l.shape))
        # exit()
        if pred.shape[-3:] != y.shape[-3:]:
            pred = F.interpolate(pred, size=y.shape[-3:], mode='trilinear', align_corners=True)
        term_aux_ce_loss = 0.05 * get_aux_ce_loss(class_wise_mask_logits=class_wise_logits, mod_target=y, select=args.select)
        term_aux_dice_loss = 0.05 * get_aux_dice_loss(class_wise_mask_logits=class_wise_logits, mod_target=y, select=args.select)
        term_seg_Dice = loss_seg_DICE.forward(pred, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(pred, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice + term_aux_ce_loss + term_aux_dice_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f, aux_ce_loss=%2.5f, aux_dice_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item(), term_aux_ce_loss.item(), term_aux_dice_loss.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        loss_aux_ce_ave += term_aux_ce_loss.item()
        loss_aux_dice_ave += term_aux_dice_loss.item()
        del x, y, name, loss
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f, ave_aux_ce_loss=%2.5f, ave_aux_dice_loss=%2.5f' 
          % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator), loss_aux_ce_ave/len(epoch_iterator), loss_aux_dice_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator),  loss_aux_ce_ave/len(epoch_iterator), loss_aux_dice_ave/len(epoch_iterator)


def validation(model, ValLoader, args):
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    
    epoch_iterator = tqdm(
        ValLoader, desc="Evaluating (X / X Steps) (loss=X.X)"
    )
    for index, batch in enumerate(tqdm(epoch_iterator)):
        # print('%d processd' % (index))
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        # label = get_tgt_with_bg(label)
        with torch.no_grad():
            pred = sliding_window_inference(inputs=image, roi_size=(args.roi_x, args.roi_y, args.roi_z), sw_batch_size=1, predictor=model)
            pred_sigmoid = F.sigmoid(pred)

        if pred_sigmoid.shape[-3:] != label.shape[-3:]:
            pred_sigmoid = F.interpolate(pred_sigmoid, size=label.shape[-3:], mode='trilinear', align_corners=True)

        torch.cuda.empty_cache()
        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_organ, recall, precision = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())
                print('val_organ: {}, recall: {}, precision:{}' .format(organ, recall, precision))
                
                dice_list[template_key][0][organ-1] += dice_organ.item()
                dice_list[template_key][1][organ-1] += 1
     
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    # if args.local_rank == 0:  
    for key in TEMPLATE.keys():
        organ_list = TEMPLATE[key]
        content = 'Task%s| '%(key)
        for organ in organ_list:
            dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
            content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
            ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
            ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1] 
    
    DSC = 0.0
    BTCV_class = 0
    for i in range(len(ORGAN_NAME)):    
        DSC += ave_organ_dice[0][i] / ave_organ_dice[1][i]
        BTCV_class += 1
    avge_DSC = DSC / BTCV_class
    print('BTCV_avg_DSC: %.4f' % avge_DSC)
    del image, label, name
    torch.cuda.empty_cache()
    return avge_DSC, ave_organ_dice 


def process(args):
    init_seed(args.seed)
    rank = 0
    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    model = SIPL(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    epoch=args.epoch,
                    backbone=args.backbone,
                    encoding=args.trans_encoding,
                    num_queries=args.num_queries,
                    t_max=args.t_max
                    )
    print('threshold_max', args.t_max)
    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])
        
    # if args.trans_encoding == 'word_embedding':
    #     word_embedding = torch.load(args.word_embedding)
    #     model.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    #     print(word_embedding.shape)
        
    model.to(args.device)
    model.train()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)

    # criterion and optimizer
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
 
    if args.backbone == 'unetpp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                              nesterov=False, weight_decay=1e-4)
    elif args.backbone == 'nnformer' or 'unetr_pp':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler, val_loader = get_loader(args)

    # if rank == 0:
    writer = SummaryWriter(log_dir='out/' + args.log_name)
    print('Writing Tensorboard logs to ', 'out/' + args.log_name)

    best_epoch = 0
    max_eval_dsc=0.0
    for m in range(args.epoch, args.max_epoch):
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce, loss_ce_aux, loss_dice_aux = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
        
        writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
        writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
        writer.add_scalar('train_aux_ce_loss', loss_ce_aux, args.epoch)
        writer.add_scalar('train_aux_dice_loss', loss_dice_aux, args.epoch)
        writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir('out/' + args.log_name):
                os.mkdir('out/' + args.log_name)
            torch.save(checkpoint, 'out/' + args.log_name + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success')
            
            DSC, organ_dice = validation(model, val_loader, args)
            torch.cuda.empty_cache()
            if max_eval_dsc < DSC:
                max_eval_dsc = DSC
                best_epoch = m
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
                torch.save(checkpoint, 'out/' + args.log_name + '/best_model.pth')
                with open('out/'+args.log_name+f'/b_val_{best_epoch}.txt', 'w') as f:
                    content = 'Average | '
                    for i in range(NUM_CLASS):
                        content += '%s: %.4f, '%(ORGAN_NAME[i], organ_dice[0][i] / organ_dice[1][i])
                        print(content)
                        f.write(content)
                        f.write('\n')
                    f.write('DSC: {}'.format(DSC))
                    f.write('\n')
                    f.write('best_epoch: {}'.format(best_epoch))
                    f.write('\n')
        args.epoch += 1



def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    #parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='swinunetr', help='The path resume from checkpoint')
    parser.add_argument('--select', default=[5,3], help='deep supervised')
    parser.add_argument('--num_queries', default=32, type=int, help='num_queries')
    parser.add_argument('--t_max', default=0.4, type=float, help='threshold')
    ## model load
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding_onehot.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=2000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=50, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_BTCV'])
    parser.add_argument('--data_root_path', default='/data/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=4, type=int, help='sample number in each ct')

    parser.add_argument('--seed',  type=int, default=2023)
    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()