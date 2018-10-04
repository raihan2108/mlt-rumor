import json


class Args:
    def __init__(self):
        self.batch_size = 15
        self.state_size = 156
        self.emb_size = 256
        self.learning_rate = 0.0005
        self.cell_type = 'lstm'
        self.rumor_label = 3
        self.stance_label = 4
        self.epochs = 30
        self.main = 'RSD'
        self.archi = 'joint'
        self.vae_latent_size = 16
        self.vae_encoder_size = 32
        self.vae_decoder_size = 32

        self.display_epoch = 1
        self.train_performnace_epoch = 1
        self.test_epoch = 1

        # self.model_type = 'mlt-us'
        self.model_type = 'mlt-bow'
        # self.model_type = 'mlt-user'
        # self.model_type = 'mlt-single'

        # self.model_type = 'bow'

    def __str__(self):
        param_dict = {
            'batch_size': self.batch_size,
            'state_size': self.state_size,
            'emb_size': self.emb_size,
            'learning_rate': self.learning_rate,
            'cell_type': self.cell_type,
            'vae_latent_size': self.vae_latent_size,
            'vae_encoder_size': self.vae_encoder_size,
            'vae_decoder_size': self.vae_decoder_size,
            'seq_len': self.seq_len,
            'vocab_size': self.vocab_size
        }

        return json.dumps(param_dict, indent=2)
