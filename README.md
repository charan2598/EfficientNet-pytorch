# EfficientNet for CIFAR-100 Classification

This repository contains an implementation of an **EfficientNet** model for image classification, specifically trained and evaluated on the **CIFAR-100** dataset. 
EfficientNet is a family of convolutional neural networks that achieve state-of-the-art accuracy while maintaining computational efficiency.


## References

1. Inverted Residual blocks were from [MobileNetV2](https://arxiv.org/abs/1801.04381) paper.
2. Squeeze and Excitation block from [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) paper.
3. Stochastic Depth block from [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) paper.
4. Official paper of [EfficientNet](https://arxiv.org/abs/1905.11946)
5. Efficient Net TensorFlow implementation for reference: [TensorFlow TPU EfficientNet Implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
6. A very good video explaining how Depthwise convolution works: [Groups, Depthwise, and Depthwise Separable Convolution](https://www.youtube.com/watch?v=vVaRhZXovbw)
7. A good blog explaining Stochastic depth [Stochastic Depth Implementation in PyTorch](https://medium.com/towards-data-science/implementing-stochastic-depth-drop-path-in-pytorch-291498c4a974)
8. Official Implementation of [Stochastic Depth / Drop Path Implementation](https://github.com/FrancescoSaverioZuppichini/DropPath/blob/main/README.ipynb) for reference.
9. Efficient Net integrated into Torchvision models: [Torchvision PyTorch EfficientNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)
