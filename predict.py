"""Loads a trained model checkpoint and makes predictions on a dataset."""

from rdkit import RDLogger

from dglt.parsing import parse_predict_args, get_newest_train_args
from dglt.train import make_predictions, write_prediction

if __name__ == '__main__':
    lg = RDLogger.logger()
    RDLogger.DisableLog('rdApp.*')
    lg.setLevel(RDLogger.CRITICAL)
    args = parse_predict_args()
    train_args = get_newest_train_args()
    avg_preds, test_smiles = make_predictions(args, train_args)
    write_prediction(avg_preds, test_smiles, args)
