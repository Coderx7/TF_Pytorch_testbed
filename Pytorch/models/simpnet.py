'''SimpNet in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class simpnet(nn.Module):
    def __init__(self, classes=10, simpnet_name='simpnet'):
        super(simpnet, self).__init__()
        self.features = self._make_layers() 
        self.classifier = nn.Linear(432, classes)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
                    name, own_state[name].size(), param.size()))

    def forward(self, x):
        #print(x.size())
        out = self.features(x)

        #Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:]) 
        #out = F.dropout2d(out, 0.02, training=True)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):

        model = nn.Sequential(
                             nn.Conv2d(3, 66, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(66, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.01),

                             nn.Conv2d(66, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.03),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.03),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.03),

                             nn.Conv2d(128, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(192, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.05),


                             nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(192, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.03),

                             nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(192, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.03),

                             nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(192, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.035),

                             nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(192, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.035),

                             nn.Conv2d(192, 288, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(288, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.05),


                             nn.Conv2d(288, 288, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(288, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.04), 

                             nn.Conv2d(288, 355, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(355, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                             #nn.Dropout2d(p=0.04), 

                             nn.Conv2d(355, 432, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(432, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),
                            )

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model



