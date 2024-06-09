from option import args
from importlib import import_module

from utils.utils import mkExpDir, save_model, calc_psnr_and_ssim_torch_metric, forward_chop
from dataset.dataset import Train_Dataset, Test_Dataset
from loss import get_loss_dict
import os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torch import optim
import torchvision
from torch.nn import functional as F



if __name__ == '__main__':
    # Setup logger
    args.save = os.path.join(args.save, args.model)
    
    logger = mkExpDir(args)
    logger.info('LOGGER SETUP COMPLETED')
    
    # Define dataset
    train_dataset = Train_Dataset(image_path=args.dir_data, 
                                patch_size=args.patch_size, 
                                scale=args.scale[0])

    train_dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.n_threads,
                            shuffle=True,)
    
    # Eval dataset
    data_test = args.data_test[0]
    test_path_HR = os.path.join(args.dir_test, data_test, f'x{args.scale[0]}', 'HR')
    test_path_LR = os.path.join(args.dir_test, data_test, f'x{args.scale[0]}', 'LR')
    
    eval_dataset = Test_Dataset(image_path_HR=test_path_HR, 
                                image_path_LR=test_path_LR)

    eval_dataloader = DataLoader(eval_dataset,
                            batch_size=1,
                            num_workers=1)
    
    logger.info('DATASET LOADED')
    
    # Define model
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    logger.info(f'Device: {device}')
    
    module = import_module('model.' + args.model.lower())
    model = module.make_model(args).to(device)
    
    # Optimizer, lr scheduler
    # vgg19 = Vgg19(requires_grad=False).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.epsilon
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(lambda x: int(x), args.decay.split('-'))),
        gamma=args.gamma
    )
    
    # Define loss function
    weight, loss_type = args.loss.split('*')
    weight = float(weight)
    loss_all = get_loss_dict(weight, loss_type, logger)
    
    # Summary Writer
    writer = SummaryWriter(log_dir=os.path.join(args.save, 'runs'))
    
    # Train
    model.to(device)

    if args.pre_train:
        data = torch.load(args.pre_train)
        if args.save_all:
            # scheduler.load_state_dict(data['scheduler'])
            
            optimizer.load_state_dict(data['opt'])
            lr = args.lr
            for milestone in list(map(lambda x: int(x), args.decay.split('-'))):
                if data['epoch'] > milestone:
                    lr *= args.gamma
            for g in optimizer.param_groups:
                g['lr'] = lr
                
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(map(lambda x: int(x), args.decay.split('-'))),
                gamma=args.gamma,
                last_epoch=data['epoch'],
            )

        model.load_state_dict(data['model'])
        logger.info(f"Load Pretrained model at epoch {data['epoch']}")
        count = data['step']
        start_epoch = data['epoch']
        log_loss = data['loss']
        max_psnr = data['eval']['max_psnr']
        max_psnr_epoch = data['eval']['max_psnr_epoch']
        max_ssim = data['eval']['max_ssim']
        max_ssim_epoch = data['eval']['max_ssim_epoch']
        logger.info('PRETRAINED MODEL LOADED')
    else:
        count = 0
        start_epoch = 0
        log_loss = []
        max_psnr = 0
        max_psnr_epoch = 0
        max_ssim = 0
        max_ssim_epoch = 0
        logger.info('USING FRESH MODEL')

    for epoch in range(start_epoch+1, args.epochs+1):
        model.train()
        for imgs in train_dataloader:
            hr, lr = imgs
            hr = hr.to(device)
            lr = lr.to(device)

            sr = model(lr)

            rec_loss = weight * loss_all['rec_loss'](sr, hr)
            loss = rec_loss

            # if epoch > args.num_init_epochs:
            #     if ('per_loss' in loss_all):
            #         sr_relu5_1 = vgg19((sr + 1.) / 2.)
            #         with torch.no_grad():
            #             hr_relu5_1 = vgg19((hr.detach() + 1.) / 2.)
            #         per_loss = args.per_w * loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
            #         loss += per_loss
            #     if ('adv_loss' in loss_all):
            #         adv_loss = args.adv_w * loss_all['adv_loss'](sr, hr)
            #         loss += adv_loss

            if count % args.print_every == 0:
                logger.info('Epoch {}/{}, Iter {}: Loss = {}, lr = {}'.format(
                    epoch,
                    args.epochs,
                    count,
                    loss.mean().item(),
                    scheduler.get_last_lr(),
                ))
                writer.flush()

            log_loss.append(loss.mean().item())
            writer.add_scalar('Loss/train', loss.mean().item(), count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

        scheduler.step()

        if epoch % args.test_every == 0:
            logger.info(f'Evaluate at epoch {epoch}!')
            model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0

                # img_eval_dir = os.path.join(args.save, 'model_eval', f'{epoch}')
                # os.mkdir(img_eval_dir)

                for imgs in eval_dataloader:
                    cnt += 1
                    hr, lr = imgs
                    hr = hr.to(device)
                    lr = lr.to(device)
                    
                    h_old = lr.size(2)
                    w_old = lr.size(3)
                       
                    if args.chop: 
                        sr = forward_chop(lr, model=model, scale=args.scale[0])
                    else:
                        sr = model(lr)

                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim_torch_metric(sr.detach(), hr.detach())

                    lr = F.interpolate(lr, (hr.size(2), hr.size(3)), mode='bicubic')

                    # torchvision.utils.save_image(torch.concat([hr, sr, lr], dim=0),
                    #                             os.path.join(img_eval_dir, f'Set5_{cnt}.png'))

                    psnr += _psnr
                    ssim += _ssim
                psnr_avg = psnr / cnt
                ssim_avg = ssim / cnt
                writer.add_scalar('PSNR/train', psnr_avg, epoch)
                writer.add_scalar('SSIM/train', ssim_avg, epoch)
                
                if psnr_avg > max_psnr:
                    max_psnr = psnr_avg
                    max_psnr_epoch = epoch
                    eval_data = {
                        'max_psnr': max_psnr,
                        'max_psnr_epoch': max_psnr_epoch,
                        'max_ssim': max_ssim,
                        'max_ssim_epoch': max_ssim_epoch,
                    }
                    save_model(args.save_all, model, optimizer, scheduler, count, epoch, log_loss, eval_data,
                            os.path.join(args.save, 'model', 'max_psnr_model.pth'))
                if ssim_avg > max_ssim:
                    max_ssim = ssim_avg
                    max_ssim_epoch = epoch
                    eval_data = {
                        'max_psnr': max_psnr,
                        'max_psnr_epoch': max_psnr_epoch,
                        'max_ssim': max_ssim,
                        'max_ssim_epoch': max_ssim_epoch,
                    }
                    save_model(args.save_all, model, optimizer, scheduler, count, epoch, log_loss, eval_data,
                            os.path.join(args.save, 'model', 'max_ssim_model.pth'))
                    
                logger.info('Eval  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                        %(max_psnr, max_psnr_epoch, max_ssim, max_ssim_epoch))
                logger.info('Eval  PSNR (current): %.3f (%d) \t SSIM (current): %.4f (%d)'
                        %(psnr_avg, epoch, ssim_avg, epoch))
                
            logger.info('Evaluation over.')
        if epoch % args.save_every == 0:
            eval_data = {
                'max_psnr': max_psnr,
                'max_psnr_epoch': max_psnr_epoch,
                'max_ssim': max_ssim,
                'max_ssim_epoch': max_ssim_epoch,
            }
            save_model(args.save_all, model, optimizer, scheduler, count, epoch, log_loss, eval_data,
                            os.path.join(args.save, 'model', 'current_model.pth'))

