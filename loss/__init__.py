# from .gan import AdversarialLoss
# from .perceptual import PerceptualLoss
from .reconstruct import ReconstructionLoss

def get_loss_dict(weight, loss_type, logger):
    loss = {}
    if (abs(weight - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss(type=loss_type)
    # if (abs(args.per_w - 0) > 1e-8):
    #     loss['per_loss'] = PerceptualLoss()
    # if (abs(args.adv_w - 0) > 1e-8):
    #     loss['adv_loss'] = AdversarialLoss(logger=logger, use_cpu=args.cpu, num_gpu=args.num_gpu,
    #         gan_type=args.GAN_type, gan_k=args.GAN_k, lr_dis=args.lr_rate_dis,
    #         train_crop_size=args.train_crop_size)
    return loss