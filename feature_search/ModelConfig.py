class ModelConfigCNN:
    def __init__(self,
                 window_size: int,
                 num_features: int,
                 output_size: int = 1,
                 num_conv: int = 3,
                 kernel_size: int = 3,
                 channels: int = 2,
                 padding: bool = False,
                 num_hidden_layers: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.1):

        self.window_size = window_size
        self.num_features = num_features
        self.output_size = output_size
        self.num_conv = num_conv
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
