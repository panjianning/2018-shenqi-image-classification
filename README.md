# 2018-shenqi-image-classification
2018 中国气象“神气”算法赛-17名, A榜：0.844， B榜：0.847

### Model
PyTorch Densenet201, 最后的线性层和最后两个denseblock可训练。

### Data augmentation
```python
trans_train = transforms.Compose([transforms.RandomResizedCrop(size=224),
                            transforms.RandomHorizontalFlip(),
#                             transforms.ColorJitter(0.5,0, 0.5,0),
                            transforms.RandomGrayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

trans_valid = transforms.Compose([transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
```
### Training
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
```

```shell
[Epoch 1] train loss 0.580791 train acc 0.798577  valid loss 0.517831 valid acc 0.812910
...................................................................................................................
[Epoch 2] train loss 0.505452 train acc 0.818829  valid loss 0.381349 valid acc 0.878556
save model...
saved.
...................................................................................................................
[Epoch 3] train loss 0.446148 train acc 0.845649  valid loss 0.384445 valid acc 0.873085
...................................................................................................................
[Epoch 4] train loss 0.403872 train acc 0.854132  valid loss 0.397824 valid acc 0.869803
...................................................................................................................
[Epoch 5] train loss 0.408331 train acc 0.853859  valid loss 0.380620 valid acc 0.868709
...................................................................................................................
[Epoch 6] train loss 0.369287 train acc 0.872742  valid loss 0.361649 valid acc 0.886214
save model...
saved.
...................................................................................................................
[Epoch 7] train loss 0.359303 train acc 0.864258  valid loss 0.395673 valid acc 0.878556
...................................................................................................................
[Epoch 8] train loss 0.331235 train acc 0.876574  valid loss 0.398124 valid acc 0.874179
...................................................................................................................
[Epoch 9] train loss 0.321326 train acc 0.884784  valid loss 0.389036 valid acc 0.874179
...................................................................................................................
[Epoch 10] train loss 0.306906 train acc 0.887247  valid loss 0.391494 valid acc 0.880744
...................................................................................................................
[Epoch 11] train loss 0.306522 train acc 0.888615  valid loss 0.399403 valid acc 0.880744
...................................................................................................................
[Epoch 12] train loss 0.273382 train acc 0.895731  valid loss 0.405565 valid acc 0.879650
...................................................................................................................
[Epoch 13] train loss 0.290727 train acc 0.898194  valid loss 0.406928 valid acc 0.873085
...................................................................................................................
[Epoch 14] train loss 0.272849 train acc 0.906678  valid loss 0.409007 valid acc 0.879650
...................................................................................................................
[Epoch 15] train loss 0.267640 train acc 0.902846  valid loss 0.410449 valid acc 0.871991
...................................................................................................................
[Epoch 16] train loss 0.272162 train acc 0.901204  valid loss 0.417731 valid acc 0.874179
...................................................................................................................
[Epoch 17] train loss 0.258703 train acc 0.906130  valid loss 0.412648 valid acc 0.875274
Finished Training
best_epoch: 6, best_val_acc 0.886214
```

在Kaggle Kernel上跑的， 训练1小时左右，内存占用3G左右。
