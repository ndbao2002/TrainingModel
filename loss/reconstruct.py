from torch import nn

class ReconstructionLoss(nn.Module):
    def __init__(self, type='L1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'L1'):
            self.loss = nn.L1Loss()
        elif (type == 'L2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)