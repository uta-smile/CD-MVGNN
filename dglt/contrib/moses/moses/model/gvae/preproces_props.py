import os
import math
from tqdm import tqdm
from multiprocessing import Process
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dglt.contrib.moses.moses.model.gvae.cvt_smiles_to_gidxs import Grammar
from dglt.contrib.moses.moses.model.gvae.model_gvae_v2 import Model as GVAEModel

# hyper-parameters
BATCH_SIZE = 64
MOSES_DIR = '/data1/jonathan/Molecule.Generation/AIPharmacist'
RULE_PATH = os.path.join(MOSES_DIR, 'moses/model/gvae/grules.txt')
DATA_DIR = '/data1/jonathan/Molecule.Generation/AIPharmacist-data'
CSV_PATH_TRN = os.path.join(DATA_DIR, 'train_w_props.csv')
CSV_PATH_TST = os.path.join(DATA_DIR, 'test_w_props.csv')
NPZ_PATH_TRN = os.path.join(DATA_DIR, 'train_w_props.npz')
NPZ_PATH_TST = os.path.join(DATA_DIR, 'test_w_props.npz')
MIN_MAX_PATH = os.path.join(DATA_DIR, 'props_min_n_max.txt')
MODEL_DIR = '/data1/jonathan/Molecule.Generation/AIPharmacist-models'
VOCAB_LOAD = os.path.join(MODEL_DIR, 'gvae.vocab')
MODEL_LOAD = os.path.join(MODEL_DIR, 'gvae.model')
CONFIG_LOAD = os.path.join(MODEL_DIR, 'gvae.config')
PROP_NAMES = ['logP', 'SA', 'NP', 'QED', 'weight']

def cvt_smiles_to_idxs(grammar, csv_data, idx_thread, csv_path):
    """Convert SMILES sequences into one-hot indices."""

    # convert SMILES sequences in a mini-batch manner
    nb_smpls = csv_data.shape[0]
    nb_mbtcs = int(math.ceil(float(nb_smpls) / BATCH_SIZE))
    print('[thread #%d] # of smiles: %d' % (idx_thread, nb_smpls))
    print('[thread #%d] # of mini-batches: %d' % (idx_thread, nb_mbtcs))
    smiles_list = csv_data['SMILES'].tolist()
    idxs_list = []
    for idx_mbtc in range(nb_mbtcs):
        idx_smpl_beg = BATCH_SIZE * idx_mbtc
        idx_smpl_end = min(idx_smpl_beg + BATCH_SIZE, nb_smpls)
        idxs_list.extend(grammar.to_idxs_batch(smiles_list[idx_smpl_beg:idx_smpl_end]))
        print('[thread #%d] progress: %d / %d' % (idx_thread, idx_mbtc + 1, nb_mbtcs))

    # write SMILES and one-hot indices to *.csv file
    csv_data_w_idxs = {key: csv_data[key].tolist() for key in csv_data.keys()}
    csv_data_w_idxs['IDXS'] = idxs_list
    data_frame = pd.DataFrame(csv_data_w_idxs)
    data_frame.to_csv(csv_path, header=(idx_thread == 0), index=False, float_format='%.4e')

class Preprocessor(object):
    """Pre-processor."""

    def __init__(self):
        """Constructor function."""

        self.model = None  # will be set in self.__restore_model()
        self.labls_min = None  # will be set in self.__calc_labls_min_max()
        self.labls_max = None  # will be set in self.__calc_labls_min_max()

    def run(self, csv_path, npz_path):
        """Pre-process *.csv file into *.npz file of features & ground-truth labels."""

        # initialize the model at the first entry
        if self.model is None:
            self.model = self.__restore_model()

        # calculate each label dimension's min/max values
        if self.labls_min is None or self.labls_max is None:
           self.__calc_labls_min_max(csv_path)

        # calculate rule indices for each SMILES sequence, and insert it into *.csv file
        csv_path_tmp_ptrn = './tmp_%02d.csv'
        csv_path_init = './tmp_init.csv'
        self.__calc_rule_idxs(csv_path, csv_path_tmp_ptrn, csv_path_init)

        # filter samples with too many grammar rules
        csv_path_filt = './tmp_filt.csv'
        self.__filter_long_seqs(csv_path_init, csv_path_filt)
        os.remove(csv_path_init)

        # extract features & ground-truth labels from *.csv file, and save into a *.npz file
        feats, labls = self.__extract_feats_n_labls(csv_path_filt)
        os.remove(csv_path_filt)
        np.savez(npz_path, feats=feats, labls=labls)

    def __restore_model(self):
        """Restore the pre-trained GVAE model."""

        model_vocab = torch.load(VOCAB_LOAD)
        model_state = torch.load(MODEL_LOAD)
        model_config = torch.load(CONFIG_LOAD)
        model = GVAEModel(model_vocab, model_config)
        model.load_state_dict(model_state)
        model = model.to('cuda')
        model.eval()

        return model

    def __calc_labls_min_max(self, csv_path):
        """Calculate each label dimension's min/max values."""

        # compute each label dimension's min/max values
        csv_data = pd.read_csv(csv_path)
        self.labls_min = np.zeros((1, len(PROP_NAMES)), dtype=np.float32)
        self.labls_max = np.zeros((1, len(PROP_NAMES)), dtype=np.float32)
        with open(MIN_MAX_PATH, 'w') as o_file:
            for idx, name in enumerate(PROP_NAMES):
                vals = np.array(csv_data[name])
                self.labls_min[0][idx] = vals.min()
                self.labls_max[0][idx] = vals.max()
                o_file.write('%.4e %.4e\n' % (self.labls_min[0][idx], self.labls_max[0][idx]))

    def __calc_rule_idxs(self, csv_path_src, csv_path_tmp_ptrn, csv_path_dst):
        """Calculate rule indices for each SMILES sequence, and insert it into *.csv file."""

        # create a grammar model for SMILES
        grammar = Grammar(RULE_PATH)

        # load SMILES sequences from *.csv file
        csv_data = pd.read_csv(csv_path_src)

        # convert SMILES sequences into one-hot indices
        nb_threads = 32
        nb_smpls = csv_data.shape[0]
        nb_smpls_per_thread = int(math.ceil(float(nb_smpls) / nb_threads))
        procs = []
        for idx_thread in range(nb_threads):
            idx_smpl_beg = nb_smpls_per_thread * idx_thread
            idx_smpl_end = min(idx_smpl_beg + nb_smpls_per_thread, nb_smpls)
            csv_data_sel = csv_data[idx_smpl_beg:idx_smpl_end]
            csv_path_tmp = csv_path_tmp_ptrn % idx_thread
            args = (grammar, csv_data_sel, idx_thread, csv_path_tmp)
            proc = Process(target=cvt_smiles_to_idxs, args=args)
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()

        # merge all *.csv files into one
        with open(csv_path_dst, 'w') as o_file:
            for idx_thread in range(nb_threads):
                csv_path_tmp = csv_path_tmp_ptrn % idx_thread
                o_file.write(open(csv_path_tmp, 'r').read())
                os.remove(csv_path_tmp)

    def __filter_long_seqs(self, csv_path_src, csv_path_dst):
        """Filter-out sequences that are too long."""

        csv_data = pd.read_csv(csv_path_src)
        idxs_list = csv_data['IDXS'].tolist()
        idxs_smpl = [idx for idx in range(len(idxs_list)) if len(idxs_list[idx].split('|')) < 150]
        print('%d vs. %d' % (len(idxs_list), len(idxs_smpl)))
        csv_data_filt = csv_data.iloc[idxs_smpl]
        csv_data_filt.to_csv(csv_path_dst, header=True, index=False, float_format='%.4e')

    def __extract_feats_n_labls(self, csv_path):
        """Extract features & ground-truth labels from *.csv file."""

        csv_data = pd.read_csv(csv_path)

        # extract features
        with torch.no_grad():
            mu_list, logvar_list = [], []
            idxs_list = csv_data['IDXS'].tolist()
            data_loader = DataLoader(idxs_list, batch_size=BATCH_SIZE, shuffle=False)
            for batch in tqdm(data_loader, desc='Extracting features'):
                tensor_list = [self.model.raw2tensor(idxs) for idxs in batch]
                mu_list_new, logvar_list_new = self.model.encode(tensor_list)
                mu_list.extend(mu_list_new)
                logvar_list.extend(logvar_list_new)
            feats = torch.cat([torch.unsqueeze(
                torch.cat([x, y]), dim=0) for x, y in zip(mu_list, logvar_list)], dim=0)
            feats = feats.cpu().numpy()
            print('{} / {}'.format(feats.shape, feats.dtype))

        # extract ground-truth labels
        eps = 1e-8
        labls = np.zeros((csv_data.shape[0], len(PROP_NAMES)), dtype=np.float32)
        for idx, name in enumerate(PROP_NAMES):
            labls[:, idx] = np.array(csv_data[name].tolist())
        labls = (labls - self.labls_min) / (self.labls_max - self.labls_min + eps)
        print('{} / {}'.format(labls.shape, labls.dtype))

        return feats, labls

def main():
    """Main entry."""

    # pre-processing
    preprocessor = Preprocessor()
    if not os.path.exists(NPZ_PATH_TRN):
        preprocessor.run(CSV_PATH_TRN, NPZ_PATH_TRN)
    if not os.path.exists(NPZ_PATH_TST):
        preprocessor.run(CSV_PATH_TST, NPZ_PATH_TST)

if __name__ == '__main__':
    main()
