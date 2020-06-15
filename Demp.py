import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.hub import load_state_dict_from_url
import copy

_all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,groups=1, width_per_group=64, replace_stride_with_dilation=None,norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = [64, 64]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        ################### RGB feature extractor Ec and Ed ####################################

        self.conv1 = nn.Conv2d(3, self.inplanes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        
        self.conv1_E = nn.Conv2d(3, self.inplanes[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_E = norm_layer(self.inplanes[1])
        self.relu_E = nn.ReLU(inplace=True)
        self.maxpool_E = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_E = self._make_layer(block, 64, layers[0], idx = 1)
        self.layer2_E = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0], idx = 1)
        self.layer3_E = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1], idx = 1)
        self.layer4_E = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2], idx = 1)
        
        ################### End RGB feature extractor Ec and Ed ####################################

        #################### Rete RGB or DEPTH only ############################
        self.avgpool_only = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_only = nn.Linear(512 * block.expansion, 1000)
        self.bn_class_only = nn.BatchNorm1d(1000)
        self.relu_class_only = nn.ReLU(inplace=True)
        self.dropout_class_only= nn.Dropout()
        self.fc_class_only = nn.Linear(1000, num_classes)
        ############## end Rete RGB or DEPTH only ###########################


        #################### Rete M ############################
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000) #poi viene aggiustata dopo
        self.bn_class = nn.BatchNorm1d(1000)
        self.relu_class = nn.ReLU(inplace=True)
        self.dropout_m= nn.Dropout()
        self.fc_class = nn.Linear(1000, num_classes)
        ############## end Rete M ###########################
        
        ################# Rete P  ########################
        self.conv11_rot = nn.Conv2d(2* 512 * block.expansion, 100, 1)
        self.bn11_rot = norm_layer(100)
        self.relu11_rot = nn.ReLU(inplace=True)
        self.conv33_rot = nn.Conv2d(100 * block.expansion, 100, 3, stride=2) #modifica: stride 2
        self.bn33_rot = norm_layer(100)
        self.relu33_rot = nn.ReLU(inplace=True)
        self.fc_p_rot = nn.Linear(3*3*100, 100) # modifica: 3*3
        self.bn_p_rot = nn.BatchNorm1d(100)
        self.relu_p_rot = nn.ReLU(inplace=True)
        self.dropout_p_rot = nn.Dropout()
        self.fc_regr_rot = nn.Linear(100, 4) #
        ################# end Rete P ########################
        
        ################# Rete P cos ########################
        self.conv11 = nn.Conv2d(2* 512 * block.expansion, 100, 1)
        self.bn11 = norm_layer(100)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(100 * block.expansion, 100, 3, stride=2)
        self.bn33 = norm_layer(100)
        self.relu33 = nn.ReLU(inplace=True)
        self.fc_p = nn.Linear(3*3*100, 100)
        self.bn_p = nn.BatchNorm1d(100)
        self.relu_p = nn.ReLU(inplace=True)
        self.dropout_p_cos = nn.Dropout()
        self.fc_regr_cos = nn.Linear(100, 1)
        ################# end Rete P cos ########################

        ################# Rete P sin ########################
        self.conv11_sin = nn.Conv2d(2* 512 * block.expansion, 100, 1)
        self.bn11_sin = norm_layer(100)
        self.relu11_sin = nn.ReLU(inplace=True)
        self.conv33_sin = nn.Conv2d(100 * block.expansion, 100, 3, stride=2)
        self.bn33_sin = norm_layer(100)
        self.relu33_sin = nn.ReLU(inplace=True)
        self.fc_p_sin = nn.Linear(3*3*100, 100)
        self.bn_p_sin = nn.BatchNorm1d(100)
        self.relu_p_sin = nn.ReLU(inplace=True)
        self.dropout_p_sin = nn.Dropout()
        self.fc_regr_sin = nn.Linear(100, 1)
        ################# end Rete P sin ########################

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, idx = 0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes[idx] != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes[idx], planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes[idx], planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes[idx] = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes[idx], planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
      
    def _forward_source_only(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool_only(x)
        x = torch.flatten(x, 1)
        x = self.fc_only(x)
        x = self.bn_class_only(x)
        x = self.relu_class_only(x)
        x = self.dropout_class_only(x)
        clas = self.fc_class_only(x)
        
        return clas

    def _forward_impl_rot(self, x, depth = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        depth = self.conv1_E(depth)
        depth = self.bn1_E(depth)
        depth = self.relu_E(depth)
        depth = self.maxpool_E(depth)

        depth = self.layer1_E(depth)
        depth = self.layer2_E(depth)
        depth = self.layer3_E(depth)
        depth = self.layer4_E(depth)

        cd = torch.cat((x, depth), 1)
        
        rot = self.conv11_rot(cd)
        rot = self.bn11_rot(rot)
        rot = self.relu11_rot(rot)
        rot =  self.conv33_rot(rot)
        rot = self.bn33_rot(rot)
        rot = self.relu33_rot(rot)
        rot = rot.view(-1, 100*3*3) 
        rot = self.fc_p_rot(rot)
        rot = self.bn_p_rot(rot)
        rot = self.relu_p_rot(rot)
        rot = self.dropout_p_rot(rot)
        res_rot = self.fc_regr_rot(rot)
        
        return res_rot

    def _forward_impl_main_task(self, x, depth = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        depth = self.conv1_E(depth)
        depth = self.bn1_E(depth)
        depth = self.relu_E(depth)
        depth = self.maxpool_E(depth)

        depth = self.layer1_E(depth)
        depth = self.layer2_E(depth)
        depth = self.layer3_E(depth)
        depth = self.layer4_E(depth)

        cd = torch.cat((x, depth), 1)
        x = self.avgpool(cd)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if setup == "tsne":
            return x
        x = self.bn_class(x)
        x = self.relu_class(x)
        x = self.dropout_m(x)
        clas = self.fc_class(x)

        return clas
    
    def _forward_impl(self, x, depth = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        depth = self.conv1_E(depth)
        depth = self.bn1_E(depth)
        depth = self.relu_E(depth)
        depth = self.maxpool_E(depth)

        depth = self.layer1_E(depth)
        depth = self.layer2_E(depth)
        depth = self.layer3_E(depth)
        depth = self.layer4_E(depth)

        cd = torch.cat((x, depth), 1)
        
        rot_cos = self.conv11(cd)
        rot_cos = self.bn11(rot_cos)
        rot_cos = self.relu11(rot_cos)
        rot_cos =  self.conv33(rot_cos)
        rot_cos = self.bn33(rot_cos)
        rot_cos = self.relu33(rot_cos)
        rot_cos = rot_cos.view(-1, 100*3*3)
        rot_cos = self.fc_p(rot_cos)
        rot_cos = self.bn_p(rot_cos)
        rot_cos = self.relu_p(rot_cos)
        rot_cos = self.dropout_p_cos(rot_cos)
        res_cos = self.fc_regr_cos(rot_cos)
        

        rot_sin = self.conv11_sin(cd)
        rot_sin = self.bn11_sin(rot_sin)
        rot_sin = self.relu11_sin(rot_sin)
        rot_sin =  self.conv33_sin(rot_sin)
        rot_sin = self.bn33_sin(rot_sin)
        rot_sin = self.relu33_sin(rot_sin)
        rot_sin = rot_sin.view(-1, 100*3*3)
        rot_sin = self.fc_p_sin(rot_sin)
        rot_sin = self.bn_p_sin(rot_sin)
        rot_sin = self.relu_p_sin(rot_sin)
        rot_sin = self.dropout_p_sin(rot_sin)
        res_sin = self.fc_regr_sin(rot_sin)
        
        return res_cos, res_sin

    

    def forward(self, rgb, depth = None, setup=None):
      if depth == None:
        return self._forward_source_only(rgb) # forward with one feature extractor
      
      if setup == "rotation_regr":
        return self._forward_impl(rgb, depth) #our variations, relative and absolute rotation
      elif setup == "standard":
        return self._forward_impl_rot(rgb, depth) # loghmani et al. implementation
      
      return self._forward_impl_main_task(rgb, depth, setup) #RGB e2e, TSNE


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)

        model.conv1_E.weight.data = copy.deepcopy(model.conv1.weight.data)
        #model.conv1_E.bias.data = copy.deepcopy(model.conv1.bias.data)
        
        model.bn1_E.weight.data = copy.deepcopy( model.bn1.weight.data)
        model.bn1_E.bias.data = copy.deepcopy(model.bn1.bias.data)
    
        
        model.layer1_E.load_state_dict(model.layer1.state_dict(), strict = False)
        model.layer2_E.load_state_dict(model.layer2.state_dict(), strict = False)
        model.layer3_E.load_state_dict(model.layer3.state_dict(), strict = False)
        model.layer4_E.load_state_dict(model.layer4.state_dict(), strict = False)
        
        model.fc_only.weight.data = copy.deepcopy(model.fc.weight.data)
        model.fc_only.bias.data = copy.deepcopy(model.fc.bias.data)
        
        torch.nn.init.xavier_uniform_(model.fc_class_only.weight)

        model.fc = nn.Linear(2 * 512 * block.expansion, 1000)
        model.fc_class = nn.Linear(1000, kwargs['num_classes'])

        torch.nn.init.xavier_uniform_(model.fc.weight)
        torch.nn.init.xavier_uniform_(model.fc_class.weight)

        torch.nn.init.xavier_uniform_(model.conv11.weight)
        #torch.nn.init.xavier_uniform_(model.bn11.weight)
        torch.nn.init.xavier_uniform_(model.conv33.weight)
        torch.nn.init.xavier_uniform_(model.fc_p.weight)
        #torch.nn.init.xavier_uniform_(model.bn33.weight)
        torch.nn.init.xavier_uniform_(model.fc_regr_cos.weight)
        
        torch.nn.init.xavier_uniform_(model.conv11_sin.weight)
        torch.nn.init.xavier_uniform_(model.conv33_sin.weight)
        torch.nn.init.xavier_uniform_(model.fc_p_sin.weight)
        #torch.nn.init.xavier_uniform_(model.bn11_sin.weight)
        #torch.nn.init.xavier_uniform_(model.bn33_sin.weight)
        torch.nn.init.xavier_uniform_(model.fc_regr_sin.weight)

        torch.nn.init.xavier_uniform_(model.conv11_rot.weight)
        torch.nn.init.xavier_uniform_(model.conv33_rot.weight)
        #torch.nn.init.xavier_uniform_(model.bn33_rot.weight)
        torch.nn.init.xavier_uniform_(model.fc_p_rot.weight)
        #torch.nn.init.xavier_uniform_(model.bn11_rot.weight)
        torch.nn.init.xavier_uniform_(model.fc_regr_rot.weight)
    return model


def Demp18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)