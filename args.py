

class Args:
    def __init__(self):
        self.batch_size = 15
        self.state_size = 256
        self.emb_size = 256
        self.learning_rate = 0.001
        self.cell_type = 'lstm'
        self.rumor_label = 2
        self.stance_label = 4
        self.epochs = 100
        self.main = 'RSD'
        self.archi = 'joint'

        self.display_epoch = 1
        self.train_performnace_epoch = 1
        self.test_epoch = 1

        # self.mode_type = 'single'
        self.mode_type = 'mlt-us'
        # self.mode_type = 'bow'