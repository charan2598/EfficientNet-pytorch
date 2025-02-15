import torch
import torch.nn as nn
import math

def get_activation_layer(activation):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif activation == "silu":
        return nn.SiLU(inplace=True)
    elif activation == "relu6":
        return nn.ReLU6(inplace=True)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ("Unknown activation function: ", activation)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        activation="relu",
        batch_norm=True,
        drop_out=False,
    ):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=(not batch_norm),
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if drop_out:
            layers.append(nn.Dropout2d(p=0.2))
        if activation is not None:
            layers.append(get_activation_layer(activation=activation))

        self.conv_layer = nn.Sequential(*layers)

    def forward(self, input):
        # print("Conv input: ", input.shape)
        output = self.conv_layer(input)
        # print("Conv output: ", output.shape)
        return output


# The inverted Residual Conv Block
class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        activation="relu",
        expansion_ratio=1,
        reduction_ratio=24,
        batch_norm=True,
        drop_out=False,
    ):
        super(MBConvBlock, self).__init__()

        expansion_layers_count = in_channels * expansion_ratio

        self.mb_conv_layers = nn.Sequential(
            # The expansion layer uses a 1x1 kernel with stride 1 just to expand the channels in the features.
            (
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=expansion_layers_count,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    activation=activation,
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                )
                if expansion_ratio > 1
                else nn.Identity()
            ),
            # Depth-wise convolution by specifying the groups param in the Conv2d.
            ConvBlock(
                in_channels=expansion_layers_count,
                out_channels=expansion_layers_count,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=expansion_layers_count,
                activation=activation,
                batch_norm=batch_norm,
                drop_out=drop_out,
            ),
            # Squeeze and Excitation Block are units added as part of MBConvBlock in the Efficient Net architecture.
            SqueezeExcitationBlock(
                channel_count=expansion_layers_count,
                activation=activation,
                reduction_ratio=reduction_ratio,
            ),
            # The opposite of expansion layer.
            ConvBlock(
                in_channels=expansion_layers_count,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=groups,
                activation=None,
                batch_norm=batch_norm,
                drop_out=drop_out,
            ),
        )

        # Stochastic Depth layer.
        self.stochastic_depth_layer = StochasticDepthBlock(survival_prob=0.99)

        # To avoid errors while adding the residual connection's input.
        self.use_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, input):
        # print("MBConv input: ", input.shape)
        if self.use_residual:
            output = self.mb_conv_layers(input) + input
        else:
            output = self.mb_conv_layers(input)
        # print("MBConv output: ", output.shape)
        return self.stochastic_depth_layer(output)


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, channel_count, activation="relu", reduction_ratio=24):
        super(SqueezeAndExcitationBlock, self).__init__()
        out_features = channel_count // reduction_ratio
        self.layers = nn.Sequential(
            # Squeeze Phase
            # Global Pooling layer: Converts CxHxW to Cx1x1 when output_size=1
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            # Excitation phase
            # FC Layer going from C to C//r
            nn.Linear(in_features=channel_count, out_features=out_features),
            get_activation_layer(activation),
            # FC Layer going from C//r to C
            nn.Linear(in_features=out_features, out_features=channel_count),
            get_activation_layer("sigmoid"),
        )

    def forward(self, input):
        batch_size, channel_size, _, _ = input.shape
        scale = self.layers(input)
        scale = scale.reshape(batch_size, channel_size, 1, 1)
        return scale * input

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channel_count, activation="relu", reduction_ratio=24):
        super(SqueezeExcitationBlock, self).__init__()
        # Squeeze up to out_featuers number of channels
        out_features = channel_count // reduction_ratio
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=channel_count, out_channels=out_features, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=out_features, out_channels=channel_count, kernel_size=1, stride=1),
            get_activation_layer(activation),
            get_activation_layer("sigmoid"),
        )

    def forward(self, input):
        batch_size, channel_size, _, _ = input.shape
        scale = self.layers(input)
        scale = scale.reshape(batch_size, channel_size, 1, 1)
        return scale * input

class StochasticDepthBlock(nn.Module):
    def __init__(self, survival_prob=0.9):
        super(StochasticDepthBlock, self).__init__()
        self.survival_prob = survival_prob

    def forward(self, input):
        if self.training:
            mask = torch.empty(size=(input.shape[0], 1, 1, 1), dtype=input.dtype, device=input.device).bernoulli_(
                p=self.survival_prob
            )
            mask = mask / self.survival_prob

            return input * mask

        return input


class EfficientNet(nn.Module):
    def __init__(self, cfg, total_classes):
        super(EfficientNet, self).__init__()
        self.config = cfg
        self.total_classes = total_classes
        self.network = self.build_network()
        self.initialise_weights()

    def build_network(self):
        layers = []
        in_channels = 3
        for layer in self.config:
            layer_type = layer[0]
            if layer_type == "ConvBlock":
                num_repeats = layer[-1]
                for i in range(num_repeats):
                    layers.append(
                        ConvBlock(
                            in_channels=in_channels,
                            out_channels=layer[5],
                            kernel_size=layer[2],
                            stride=layer[3],
                            padding=layer[4],
                            groups=1,
                            activation="silu",
                            batch_norm=True,
                            drop_out=False,
                        )
                    )
                    in_channels = layer[5]
            elif layer_type == "MBConvBlock":
                if layer[-1] == 1:
                    layers.append(
                        MBConvBlock(
                            in_channels=in_channels,
                            out_channels=layer[5],
                            kernel_size=layer[2],
                            stride=layer[3],
                            padding=layer[4],
                            groups=1,
                            activation="silu",
                            expansion_ratio=layer[1],
                            reduction_ratio=layer[6],
                            batch_norm=True,
                            drop_out=False,
                        )
                    )
                    in_channels = layer[5]
                else:
                    layers.append(
                        MBConvBlock(
                            in_channels=in_channels,
                            out_channels=layer[5],
                            kernel_size=layer[2],
                            stride=layer[3],
                            padding=layer[4],
                            groups=1,
                            activation="silu",
                            expansion_ratio=layer[1],
                            reduction_ratio=layer[6],
                            batch_norm=True,
                            drop_out=False,
                        )
                    )
                    in_channels = layer[5]
                    for i in range(layer[-1] - 2):
                        layers.append(
                            MBConvBlock(
                                in_channels=in_channels,
                                out_channels=layer[5],
                                kernel_size=layer[2],
                                stride=1,
                                padding=layer[4],
                                groups=1,
                                activation="silu",
                                expansion_ratio=layer[1],
                                reduction_ratio=layer[6],
                                batch_norm=True,
                                drop_out=False,
                            )
                        )
                        in_channels = layer[5]
                    layers.append(
                        MBConvBlock(
                            in_channels=in_channels,
                            out_channels=layer[5],
                            kernel_size=layer[2],
                            stride=1,
                            padding=layer[4],
                            groups=1,
                            activation="silu",
                            expansion_ratio=layer[1],
                            reduction_ratio=layer[6],
                            batch_norm=True,
                            drop_out=False,
                        )
                    )
            elif layer_type == "PoolingBlock":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=layer[2], stride=layer[3], padding=layer[4]
                    )
                )
            elif layer_type == "FCLayer":
                layers.extend(
                    [
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(in_features=1280, out_features=self.total_classes),
                    ]
                )
            else:
                raise ("Unknown Layer name: ", layer_type)

        return nn.Sequential(*layers)

    def initialise_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init_range = 1.0 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
                nn.init.zeros_(module.bias)

    def forward(self, input):
        return self.network(input)


def test_ConvBlock():
    block = ConvBlock(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        activation="silu",
        batch_norm=True,
        drop_out=True,
    )
    sample_input = torch.randn(64, 3, 32, 32)
    output = block(sample_input)

    print("Output shape of ConvBlock: ", output.shape)


def test_MBConvBlock():
    block = MBConvBlock(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        activation="silu",
        expansion_ratio=1,
        reduction_ratio=24,
        batch_norm=True,
        drop_out=True,
    )
    sample_input = torch.randn(64, 3, 32, 32)
    output = block(sample_input)

    print("Output shape of MBConvBlock: ", output.shape)


def test_SqueezeAndExcitationBlock():
    block = SqueezeAndExcitationBlock(channel_count=64, reduction_ratio=24)
    sample_input = torch.randn(64, 64, 32, 32)
    output = block(sample_input)

    print("Output shape of SqueezeAndExcitationBlock: ", output.shape)


def test_StochasticDepthBlock():
    block = StochasticDepthBlock(survival_prob=0.5)
    sample_input = torch.randn(64, 3, 32, 32)
    output = block(sample_input)

    print("Output shape of StochasticDepthBlock: ", output.shape)


def test_EfficientNet():
    model = EfficientNet(baseline_network_config, 100)
    sample_input = torch.randn(64, 3, 32, 32)
    output = model(sample_input)

    print("Output shape of EfficientNet: ", output.shape)


def test_OriginalEfficientNet():
    model = EfficientNet(original_baseline_network_cfg, 100)
    sample_input = torch.randn(64, 3, 224, 224)
    output = model(sample_input)

    print("Output shape of Original EfficientNet: ", output.shape)


baseline_network_config = [
    # [Block, expansion_ratio, kernel, stride, padding, channels, reduction_ratio, layers]
    ["ConvBlock", 1, 3, 2, 1, 32, 1, 1],
    ["MBConvBlock", 1, 3, 2, 1, 16, 4, 1],
    ["MBConvBlock", 6, 3, 1, 1, 24, 24, 2],
    ["MBConvBlock", 6, 5, 1, 2, 40, 24, 2],
    ["MBConvBlock", 6, 3, 1, 1, 80, 24, 3],
    ["MBConvBlock", 6, 5, 1, 2, 112, 24, 3],
    ["MBConvBlock", 6, 5, 1, 2, 192, 24, 4],
    ["MBConvBlock", 6, 3, 2, 1, 320, 24, 1],
    ["ConvBlock", 1, 3, 1, 1, 1280, 1, 1],
    ["PoolingBlock", -1, 2, 1, 0, -1, -1, -1, -1],
    ["FCLayer"],
]

original_baseline_network_cfg = [
    ["ConvBlock", 1, 3, 2, 1, 32, 1, 1],
    ["MBConvBlock", 1, 3, 1, 1, 16, 4, 1],
    ["MBConvBlock", 6, 3, 2, 1, 24, 24, 2],
    ["MBConvBlock", 6, 5, 2, 2, 40, 24, 2],
    ["MBConvBlock", 6, 3, 2, 1, 80, 24, 3],
    ["MBConvBlock", 6, 5, 1, 2, 112, 24, 3],
    ["MBConvBlock", 6, 5, 2, 2, 192, 24, 4],
    ["MBConvBlock", 6, 3, 1, 1, 320, 24, 1],
    ["ConvBlock", 1, 1, 1, 1, 1280, 1, 1],
    # ["PoolingBlock", -1, 2, 1, 0, -1, -1, -1, -1],
    ["FCLayer"],
]

if __name__ == "__main__":
    # Unit Testing
    # test_ConvBlock()
    # test_SqueezeAndExcitationBlock()
    # test_StochasticDepthBlock()
    # test_MBConvBlock()

    # # Integration Testing
    # test_EfficientNet()
    test_OriginalEfficientNet()
    model = EfficientNet(original_baseline_network_cfg, 100)
    print(model)
