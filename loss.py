import torch
import torch.nn as nn
import torch.nn.functional as F

from knn_cuda import KNN
import auction_match_cupy
import gather_cupy


class RepulsionLoss(torch.nn.Module):
    def __init__(self, num):
        super(RepulsionLoss, self).__init__()
        self.k = num
        self.knn_repulsion = KNN(k=self.k, transpose_mode=True)

    def forward(self, x, h=0.0005): # h = 0.0005
        # x : (B,N,k)
        dist, idx = self.knn_repulsion(x, x)
        # print(dist)
        dist = dist[:, :, 1:self.k] ** 2
        loss = torch.clamp(-dist+h, min=0)
        loss = torch.mean(loss)

        return loss

class Loss_fn(torch.nn.Module):
    def __init__(self, lambda_cd=1.0, lambda_rl=1.0, k=10):
        super(Loss_fn, self).__init__()

        self.lambda_cd = lambda_cd
        self.lambda_rl = lambda_rl
        self.k = k


        def get_emd_loss(pred,gt,radius=1.0):
            '''
            pred and gt is B N 3
            '''
            idx, _ = auction_match_cupy.AuctionMatch()(pred.contiguous(), gt.contiguous())
            #gather operation has to be B 3 N
            #print(gt.transpose(1,2).shape)
            matched_out = gather_cupy.GatherOperation()(gt.transpose(1, 2).contiguous(), idx)
            matched_out = matched_out.transpose(1, 2).contiguous()
            dist2 = (pred - matched_out) ** 2
            dist2 = dist2.view(dist2.shape[0], -1)  # <-- ???
            dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
            dist2 /= radius
            return torch.mean(dist2)

        # self.CD = ChamferLoss()
        self.CD = get_emd_loss
        # self.CD = dist_emd.earth_mover_distance
        self.RL = RepulsionLoss(self.k)

    def forward(self, x, gt):

        loss_cd = self.CD(x, gt)
        loss_rl = self.RL(x)

        return loss_cd * self.lambda_cd, loss_rl * self.lambda_rl


if __name__ == "__main__":
    a = torch.randn(2, 50, 3).cuda()
    b = torch.randn(2, 50, 3).cuda()

    model = Loss_fn().cuda()
    print(model(a, b))

