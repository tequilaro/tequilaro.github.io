---
layout:     post
title:      "pytorch DataParallel 多GPU使用"
#subtitle:     人群密度模型示例
tags:
    - Pytorch
---

**单GPU：**

```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

**多GPU：**

```
device_ids = [0,1,2,3]
```

```
model = model.cuda(device_ids[0])
model = nn.DataParallel(model, device_ids=device_ids)
```

```
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
```

```
optimizer.module.step()
```

```
for param_lr in optimizer.module.param_groups:  # 同样是要加module
        #     param_lr['lr'] = param_lr['lr'] * 0.999
```


**加载多GPU预训练模型**

```
model = ft_net()
pretained_model = torch.load('./model/all/8_model.pkl')
pretained_dict = pretained_model.module.state_dict()
model = ft_net()
model.load_state_dict(pretained_dict)
```
