import torch.nn as nn

try:
    import torchvision

    cnnlib = {
        "inception_v3": torchvision.models.inception_v3,
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
    }

except ModuleNotFoundError:
    msg = (
        "If using CNN, it requires torchvision.\n"
        "Please install extra packages as follows:\n"
        "  poetry install -extras torchvision"
    )
    print(msg)


class CNN(nn.Module):
    def __init__(self, n_units, cnn_arch="resnet18", disable_update=False):
        super(CNN, self).__init__()
        if cnn_arch == "inception_v3":
            cnn = cnnlib[cnn_arch](pretrained=True, aux_logits=False)
        else:
            cnn = cnnlib[cnn_arch](pretrained=True)
        if disable_update:
            for p in cnn.parameters():
                p.requires_grad = False
        cnn.fc = nn.Linear(cnn.fc.in_features, n_units, bias=False)
        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)

    def fix_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("disable cnn update.")
