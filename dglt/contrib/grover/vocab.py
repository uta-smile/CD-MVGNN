import pickle
from collections import Counter
from multiprocessing import Pool

import tqdm
from rdkit import Chem

from dglt.contrib.grover.mol2features import atom_to_vocab


class TorchVocab(object):
    """
    Defines the vocabulary for atoms in molecular.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<other>']):
        """

        :param counter:
        :param max_size:
        :param min_freq:
        :param specials:
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.other_index = 1
        self.pad_index = 0

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                self.freqs[w] = 0
            self.freqs[w] += v.freqs[w]

    def mol_to_seq(self, mol, with_len=False):
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        seq = [self.stoi.get(atom_to_vocab(mol, atom), self.other_index) for i, atom in enumerate(mol.GetAtoms())]
        return (seq, len(seq)) if with_len else seq

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class AtomVocab(TorchVocab):
    def __init__(self, smiles, max_size=None, min_freq=1):
        print("Building atom vocab. Smiles: %d" % (len(smiles)))
        counter = Counter()

        for smi in tqdm.tqdm(smiles):
            mol = Chem.MolFromSmiles(smi)
            for i, atom in enumerate(mol.GetAtoms()):
                v = atom_to_vocab(mol, atom)
                counter[v] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def __init__(self, file_path, max_size=None, min_freq=1, num_workers=1, total_lines=None):
        print("Building atom vocab. Filepath: %s" % (file_path))

        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        if total_lines is None:
            def file_len(fname):
                with open(fname) as f:
                    for i, l in enumerate(f):
                        pass
                return i + 1

            total_lines = file_len(file_path)

        counter = Counter()
        pbar = tqdm.tqdm(total=total_lines)
        pool = Pool(num_workers)
        res = []
        batch = 50000
        callback = lambda a: pbar.update(batch)
        for i in range(int(total_lines / batch + 1)):
            start = int(batch * i)
            end = min(total_lines, batch * (i + 1))
            # print("Start: %d, End: %d"%(start, end))
            res.append(pool.apply_async(AtomVocab.read_smiles_from_file,
                                        args=(file_path, start, end,),
                                        callback=callback))
            # read_smiles_from_file(lock, file_path, start, end)
        pool.close()
        pool.join()
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]

        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    @staticmethod
    def read_smiles_from_file(file_path, start, end):
        # print("start")
        smiles = open(file_path, "r")
        smiles.readline()
        sub_counter = Counter()
        for i, smi in enumerate(smiles):
            if i < start:
                continue
            if i >= end:
                break
            mol = Chem.MolFromSmiles(smi)
            for i, atom in enumerate(mol.GetAtoms()):
                v = atom_to_vocab(mol, atom)
                sub_counter[v] += 1
        # print("end")
        return sub_counter

    @staticmethod
    def load_vocab(vocab_path: str) -> 'AtomVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--vocab_save_path', required=False, type=str)
    parser.add_argument('--vocab_max_size', type=int, default=None)
    parser.add_argument('--vocab_min_freq', type=int, default=1)
    args = parser.parse_args()

    # fin = open(args.data_path, 'r')
    # lines = fin.readlines()

    # atomvocab = AtomVocab(smiles=lines, max_size=args.vocab_max_size, min_freq=args.vocab_min_freq)
    atomvocab = AtomVocab(file_path=args.data_path,
                          max_size=args.vocab_max_size,
                          min_freq=args.vocab_min_freq,
                          num_workers=20)
    print("\nVOCAB SIZE:", len(atomvocab))
    atomvocab.save_vocab(args.vocab_save_path)

# build()

# def load_vocab(vocab_load_path="../drug_data/all_smi_vocab.pkl"):
#    atomvocab = AtomVocab.load_vocab(vocab_load_path)
    # counter = atomvocab.freqs
    # words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    # words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    # fout = open("../drug_data/all_smi_vocab_by_element_stat.txt", "w")
    # for word, freq in words_and_frequencies:
    #    fout.write("%s\t%d\n" % (word, freq))
    # fout.close()
#    return atomvocab

# build()
#load_vocab()
# load("../drug_data/all_smi_vocab_by_element.pkl")
