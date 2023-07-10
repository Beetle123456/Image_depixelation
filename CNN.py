import torch


class CNN(torch.nn.Module):

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 num_av_pool: int,
                 kernel_size: int = 3,
                 activation_function: torch.nn.Module = torch.nn.ReLU(),
                 ):

        super().__init__()
        self.input = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_av_pool = num_av_pool
        self.activation_function = activation_function


        hidden_layers = []
        flag = 0
        for layers in range(num_layers):
            conv_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels,
                                         kernel_size=kernel_size, padding=kernel_size//2)
            bn_layer = torch.nn.BatchNorm2d(num_features=hidden_channels)
            average_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            input_channels = hidden_channels
            hidden_layers.append(conv_layer)
            hidden_layers.append(bn_layer)
            hidden_layers.append(activation_function)
            if flag < num_av_pool:
                hidden_layers.append(average_pool)
                hidden_channels *= 2
                flag += 1

        self.model = torch.nn.Sequential(*hidden_layers)    # for automatic parameter registration
        self.output_layer = torch.nn.Conv2d(in_channels=hidden_channels, out_channels=1,
                                            kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = self.model(x)
        x = self.output_layer(x)
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        # make the tensor to of size 64,64 again, uses interpolation and fills in missing values by taking weighted
        # average values of neighboring pixels
        return x


"""
Produces as many Conv_layers followed by batch_normalization layers as i want, if i think i don't have enough layers i 
can simply add more layers and maybe change the kernel size or so, to see if that works.
Batch normalization should help with training speed and should reduce overfitting by normalizing the activations,
each layer will not "depend" as much to the previous layer when training. Gradients can flow more "smoothly".
"""