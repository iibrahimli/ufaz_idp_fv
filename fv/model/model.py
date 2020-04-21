import torch
import torch.nn as nn
import torchvision.models as models


class face_resnet(nn.Module):

    def __init__(self, embedding_dimension=128, pretrained=False):
        """
        Constructs a ResNet-34 model for FaceNet training using triplet loss.

        Args:
            embedding_dimension (int): required dimension of the resulting 
                embedding layer that is outputted by the model. Using triplet
                loss. Defaults to 128.
            pretrained (bool): if True, returns a model pre-trained on the
                ImageNet dataset from a PyTorch repository. Defaults to False.

        """

        super(face_resnet, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)


    def l2_norm(self, inp):
        """
        Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository
        https://github.com/liorshk/facenet_pytorch/blob/master/model.py

        """
        
        input_size = inp.size()
        buffer = torch.pow(inp, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(inp, norm.view(-1, 1).expand_as(inp))
        output = _output.view(input_size)

        return output


    def forward(self, images):
        """
        Forward pass to output the embedding vector (feature vector)
        after l2-normalization and multiplication by scalar (alpha)
        
        """
        
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        # multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        # equation 9: number of classes in VGGFace2 dataset = 9131
        # lower bound on alpha = 5, multiply alpha by 2; alpha = 10
        alpha = 10
        embedding = embedding * alpha

        return embedding