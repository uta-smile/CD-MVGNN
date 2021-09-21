from dglt.contrib.moses.moses.model.vae import VAE, VAETrainer, vae_parser, VAEGenerator, vae_generate_parser
from dglt.contrib.moses.moses.model.organ import ORGAN, ORGANTrainer, ORGANGenerator, \
    organ_train_parser, organ_generate_parser
from dglt.contrib.moses.moses.model.aae import AAE, AAETrainer, aae_parser
from dglt.contrib.moses.moses.model.char_rnn import CharRNN, CharRNNTrainer, char_rnn_parser
from dglt.contrib.moses.moses.model.sd_vae import SDVAE, SDAutoEncoder, SDVAETrainer, SDVAEGenerator, sdvae_parser
from dglt.contrib.moses.moses.model.gvae import GVAE, GVAETrainer, gvae_parser, GVAEGenerator, gvae_generate_parser
from dglt.contrib.moses.moses.model.jt_vae import JTVAE, JTVAETrainer, jtvae_parser, JTVAEGenerator, jtvae_generate_parser
from dglt.contrib.moses.moses.model.mpnn import MPGVAE, MPGVAETrainer, mpgvae_parser, mpgvae_config


class ModelsStorage():

    def __init__(self):
        self._models = {}
        self.add_model('aae', AAE, AAETrainer, aae_parser)
        self.add_model('char_rnn', CharRNN, CharRNNTrainer, char_rnn_parser)
        self.add_model('vae', VAE, VAETrainer, vae_parser, VAEGenerator, vae_generate_parser)
        self.add_model('sdvae', SDVAE, SDVAETrainer, sdvae_parser, SDVAEGenerator, sdvae_parser)
        self.add_model('organ', ORGAN, ORGANTrainer, organ_train_parser,
                       ORGANGenerator, organ_generate_parser)
        self.add_model('gvae', GVAE, GVAETrainer, gvae_parser, GVAEGenerator, gvae_generate_parser)
        self.add_model('jtvae', JTVAE, JTVAETrainer, jtvae_parser, JTVAEGenerator, jtvae_generate_parser)
        self.add_model('mpgvae', MPGVAE, MPGVAETrainer, mpgvae_parser, modify_config_=mpgvae_config)

    def add_model(self, name, class_, trainer_, parser_,
                  generator_=None, generate_parser_=None, modify_config_=None):
        self._models[name] = {
            'class': class_,
            'trainer': trainer_,
            'parser': parser_,
            'generator': generator_,
            'generate_parser': generate_parser_,
            'modify_config': modify_config_,
        }

    def get_model_names(self):
        return list(self._models.keys())

    def get_model_trainer(self, name):
        return self._models[name]['trainer']

    def get_model_reconstructor(self, name):
        return self._models[name]['reconstructor']

    def get_model_class(self, name):
        return self._models[name]['class']

    def get_model_train_parser(self, name):
        return self._models[name]['parser']

    def get_model_generator(self, name):
        return self._models[name]['generator']

    def get_model_generate_parser(self, name):
        return self._models[name]['generate_parser']

    def get_modify_config(self, name):
        return self._models[name]['modify_config']
