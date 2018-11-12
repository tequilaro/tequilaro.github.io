---
layout:     post
title:      "pytorch权重初始化"
tags:
    - Pytorch
---

**[PyTorch官网初始化文档](https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn%20functional#torch-nn-init)**

本人目前刚从caffe转pytorch，由于pytorch灵活的特性和其他框架所没有的动态图特性，最近要出1.0版本与Caffe2结合，使学术与落地工程无缝对接。

实验发现pytorch使用，必须对权重初始化，否则某些情况下的损失无法收敛。

在使用大多如下使用：

```
def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
```

```
model = C3D()
model.apply(weights_init)
```

**torch.nn.init有如下几种（0.4版本，目前最新）：**

torch.nn.init.calculate_gain(nonlinearity, param=None)

```
gain = nn.init.calculate_gain('leaky_relu')
```

torch.nn.init.uniform_(tensor, a=0, b=1)
 
```
w = torch.empty(3, 5)
nn.init.uniform_(w)
```

torch.nn.init.normal_(tensor, mean=0, std=1)
 
```
w = torch.empty(3, 5)
nn.init.normal_(w)
```

torch.nn.init.constant_(tensor, val) 
 
```
torch.empty(3, 5)
nn.init.constant_(w, 0.3)
```

torch.nn.init.eye_(tensor)
 
```
w = torch.empty(3, 5)
nn.init.eye_(w)
```

torch.nn.init.dirac_(tensor) 

```
w = torch.empty(3, 16, 5, 5)
nn.init.dirac_(w)
```

torch.nn.init.xavier_uniform_(tensor,gain=1) 

```
w = torch.empty(3, 5)
nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
```

torch.nn.init.xavier_normal_(tensor, gain=1)

```
w = torch.empty(3, 5)
nn.init.xavier_normal_(w)
```

torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

```
w = torch.empty(3, 5)
nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
```

torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

```
w = torch.empty(3, 5)
nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
```

torch.nn.init.orthogonal_(tensor, gain=1)

```
w = torch.empty(3, 5)
nn.init.orthogonal_(w)
```

torch.nn.init.sparse_(tensor, sparsity, std=0.01)

```
w = torch.empty(3, 5)
nn.init.sparse_(w, sparsity=0.1)
```

