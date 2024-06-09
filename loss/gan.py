import torch
from torch import optim
from torch import nn
from utils.utils import conv3x3

class Discriminator(nn.Module):
    def __init__(self, in_size=160):
        super(Discriminator, self).__init__()
        self.conv1 = conv3x3(3, 32)
        self.LReLU1 = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(32, 32, 2)
        self.LReLU2 = nn.LeakyReLU(0.2)
        self.conv3 = conv3x3(32, 64)
        self.LReLU3 = nn.LeakyReLU(0.2)
        self.conv4 = conv3x3(64, 64, 2)
        self.LReLU4 = nn.LeakyReLU(0.2)
        self.conv5 = conv3x3(64, 128)
        self.LReLU5 = nn.LeakyReLU(0.2)
        self.conv6 = conv3x3(128, 128, 2)
        self.LReLU6 = nn.LeakyReLU(0.2)
        self.conv7 = conv3x3(128, 256)
        self.LReLU7 = nn.LeakyReLU(0.2)
        self.conv8 = conv3x3(256, 256, 2)
        self.LReLU8 = nn.LeakyReLU(0.2)
        self.conv9 = conv3x3(256, 512)
        self.LReLU9 = nn.LeakyReLU(0.2)
        self.conv10 = conv3x3(512, 512, 2)
        self.LReLU10 = nn.LeakyReLU(0.2)

        self.fc1 = nn.Linear(in_size//32 * in_size//32 * 512, 1024)
        self.LReLU11 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.LReLU1(self.conv1(x))
        x = self.LReLU2(self.conv2(x))
        x = self.LReLU3(self.conv3(x))
        x = self.LReLU4(self.conv4(x))
        x = self.LReLU5(self.conv5(x))
        x = self.LReLU6(self.conv6(x))
        x = self.LReLU7(self.conv7(x))
        x = self.LReLU8(self.conv8(x))
        x = self.LReLU9(self.conv9(x))
        x = self.LReLU10(self.conv10(x))

        x = x.view(x.size(0), -1)
        x = self.LReLU11(self.fc1(x))
        x = self.fc2(x)

        return x
    
class AdversarialLoss(nn.Module):
    def __init__(self, logger, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1,
        lr_dis=1e-4, train_crop_size=40):

        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = Discriminator(train_crop_size*4).to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])

    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict