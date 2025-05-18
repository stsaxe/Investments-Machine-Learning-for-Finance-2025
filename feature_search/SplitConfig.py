class SplitConfig:
    def __init__(self,
                 window_size: int,
                 prediction_length: int,
                 look_ahead: int,
                 val_split: float,
                 test_split: float,
                 is_indexed: bool,
                 scale_target: bool,
                 fixed_feature_size: int,
                 batch_size: int):

        self.window_size = window_size
        self.prediction_length = prediction_length
        self.look_ahead = look_ahead
        self.val_split = val_split
        self.test_split = test_split
        self.is_indexed = is_indexed
        self.scale_target = scale_target
        self.fixed_feature_size = fixed_feature_size
        self.batch_size = batch_size