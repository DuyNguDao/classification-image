"""
Member: DA0 DUY NGU, LE VAN THIEN
"""
from utils.evaluate import evaluate_cm
import argparse
from utils.load_model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument("--folder-test", help="folder file image", default='', type=str)
    parser.add_argument("--folder-model", help="folder contain file model", default='', type=str)
    parser.add_argument("--path-save", help="path save", default='', type=str)
    args = parser.parse_args()
    path_test = args.folder_test
    save = args.path_save
    # load model
    model = Model(args.folder_model)
    print('Start evaluate model...')
    evaluate_cm(model=model, test_set_path=path_test, save_cm=save, normalize=False, show=True)
    print('Finish.')
