

class Args:
    def __init__(self):
        self.batch_size = 15
        self.state_size = 256
        self.emb_size = 256
        self.learning_rate = 0.0005
        self.cell_type = 'gru'
        self.rumor_label = 3
        self.stance_label = 4
        self.epochs = 50
        self.main = 'RSD'
        self.archi = 'joint'
        self.vae_latent_size = 16
        self.vae_encoder_size = 64
        self.vae_decoder_size = 64

        self.display_epoch = 1
        self.train_performnace_epoch = 1
        self.test_epoch = 1

        # self.model_type = 'mlt-us'
        # self.model_type = 'mlt-bow'
        # self.model_type = 'mlt-user'

        # self.model_type = 'single'
