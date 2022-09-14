import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_features, kernel_size=3, hidden_features=None, out_features=None,
                 norm_layer=nn.LayerNorm, act_layer=nn.ReLU, group=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        self.norm1 = norm_layer(hidden_features)

        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False, groups=group)
        self.act2 = act_layer()
        self.norm2 = norm_layer(hidden_features)

        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)
        self.act3 = act_layer()
        self.norm3 = norm_layer(out_features)

    def forward(self, x):
        short_cut = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x + short_cut)
        return x

class SuperColor(nn.Module):
    def __init__(self, block_num=2, input_dim=16, kernel_size=3, group=4):
        super().__init__()
        image_dim = 9
        self.embedding = nn.Conv2d(image_dim, input_dim, 3, stride=1, groups=1, padding=1, bias=True)
        self.blocks = nn.ModuleList([
            ConvBlock(input_dim, kernel_size=kernel_size, hidden_features=input_dim//2, group=group, out_features=16, norm_layer=nn.BatchNorm2d)
            for i in range(block_num)])
        self.embedding_decoder = nn.Conv2d(16, 3, 3, stride=1, groups=1, padding=1, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, img_l_interp,gray_h,color_mask):
        x = torch.cat([img_l_interp, gray_h],dim=1)
        x = torch.cat([x, color_mask], dim=1)
        x = self.embedding(x)
        for b in self.blocks:
            x = b(x)
        x = self.embedding_decoder(x)
        return x

    def forward_loss(self, pred, gray_h, img_h, alpha=0.0):
        loss_l2 = (pred + gray_h - img_h) ** 2
        loss_l2 = loss_l2.mean()
        loss_l1 = torch.abs((pred + gray_h - img_h))
        loss_l1 = loss_l1.mean()
        return loss_l2 * (1.0 - alpha) + loss_l1 * alpha




