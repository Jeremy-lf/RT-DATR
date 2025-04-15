import paddle
import paddle.nn as nn
from paddle.autograd import PyLayer
import numpy as np
import cv2


class GradientReversalFunction(PyLayer):
    """Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_=1):
        """Forward in networks
        """
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        """Backward in networks
        """
        lambda_, = ctx.saved_tensor()
        dx = -lambda_ * grads * 0.1
        return dx


class GradientReversalLayer(nn.Layer):
    """Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    def __init__(self, lambda_=1):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """Forward in networks
        """
        return GradientReversalFunction.apply(x, self.lambda_)


class _ImageDA(nn.Layer):
    def __init__(self, dim):
        super(_ImageDA, self).__init__()
        self.dim = dim  # feat layer
        self.Conv1 = nn.Conv2D(self.dim, 512, kernel_size=1, bias_attr=False)
        self.Conv2 = nn.Conv2D(512, 2, kernel_size=1, bias_attr=False)   
        self.reLu = nn.ReLU()
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self.grad_layer = GradientReversalLayer()

    def forward(self, x, need_backprop):
        x = self.grad_layer(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        label = self.LabelResizeLayer(x, need_backprop)
        return x, label


class ImageLabelResizeLayer(nn.Layer):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()

    def forward(self, x, need_backprop):
        feats = x.detach().numpy() # 将Tensor转换为NumPy数组，但不保留梯度  
        lbs = need_backprop.detach().numpy()
        gt_blob = np.zeros((lbs.shape[0], feats.shape[2], feats.shape[3], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            lb = np.array([lbs[i]])
            lbs_resize = cv2.resize(lb, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)
            gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

        channel_swap = (0, 3, 1, 2)
        gt_blob = gt_blob.transpose(channel_swap)
        y = paddle.to_tensor(gt_blob)
        y = y.squeeze(1).astype('int64')
        return y
