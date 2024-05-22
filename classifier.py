import torch
import torch.nn as nn
from collections import OrderedDict#
class Classifier(nn.Module):
    def __init__(self,alph,feature_size,num_classes, use_gpu=True):
        super(Classifier, self).__init__()
        #self.sketch_feature = sketch_feature
        #self.view_feature = view_feature
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.feature_size = feature_size
        self.alph = alph
        self.embedding_dim = 256

        #self.fc1 = nn.Sequential(nn.Linear(self.feature_size, num_classes))
        self.fc1 = nn.Sequential(nn.Linear(self.feature_size, 1024),nn.BatchNorm1d(1024,eps=2e-5),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 512),nn.BatchNorm1d(512,eps=2e-5),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(512, 256),nn.BatchNorm1d(256,eps=2e-5),nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(256, 128),nn.BatchNorm1d(128,eps=2e-5))

        self.fc5 = TransformEmbeddingToLogit(in_features=128,
                                            out_features=self.num_classes,
                                            embedding_normalization=True,
                                            weight_normalization=True)


    def forward(self,x):
        x1 = self.fc1(x)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        x1 = self.fc4(x1)
        x1 = nn.functional.normalize(x1, dim=1)
        logits,weight = self.fc5(x1)
        return x1,logits,weight

class TransformEmbeddingToLogit(nn.Module):
    r"""Transform embeddings to logits via a weight projection, additional normalization supported
    Applies a matrix multiplication to the incoming data.

    Without normalization: :math:`y = xW`;

    With weight normalization: :math:`w=x\cdot\frac{W}{\lVert W\rVert}`;

    With embedding normalization: :math:`w=\frac{x}{\lVert x\rVert}\cdot W`;

    With weight and embedding normalization: :math:`w=\frac{x}{\lVert x\rVert}\cdot\frac{W}{\lVert W\rVert}`.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        embedding_normalization (bool): whether or not to l2 normalize the embeddings. Default: `False`
        weight_normalization (bool): whether or not to l2 normalize the weight. Default: `False`

    Shape:
        - Input: :math:`(N, C_{in})` where :math:`C_{in} = \text{in\_features}`
        - Output: :math:`(N, C_{out})` where :math:`C_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{in\_features}, \text{out\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = TransformEmbeddingToLogit(20, 30, embeding_normalization=True, weight_normalization=True)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding_normalization: bool = False,
        weight_normalization: bool = False,
    ) -> None:
        super(TransformEmbeddingToLogit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.embedding_normalization = embedding_normalization
        self.weight_normalization = weight_normalization
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        print(torch.Tensor(in_features, out_features).dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.embedding_normalization:
            x = nn.functional.normalize(x, dim=1)
        if self.weight_normalization:
            weight = nn.functional.normalize(weight, dim=0)
        logits = x.matmul(weight)
        return logits,weight

    def extra_repr(self) -> str:
        return (
            "in_features={in_features}, out_features={out_features}, embedding_normalization={embedding_normalization}, "
            "weight_normalization={weight_normalization}".format(**self.__dict__)
        )
