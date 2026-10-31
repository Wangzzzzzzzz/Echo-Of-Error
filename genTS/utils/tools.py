from logging import basicConfig
import matplotlib.pyplot as plt
import torch
import os
import shutil
import inspect
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2, alpha=0.5)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, alpha=0.5)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def save_copy_of_files(checkpoint_callback):
    """Copy from TSLANet"""
    # Get the frame of the caller of this function
    caller_frame = inspect.currentframe().f_back

    # Get the filename of the caller
    caller_filename = caller_frame.f_globals["__file__"]

    # Get the absolute path of the caller script
    caller_script_path = os.path.abspath(caller_filename)

    # Destination directory (PyTorch Lightning saving directory)
    destination_directory = checkpoint_callback.dirpath

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Copy the caller script to the destination directory
    shutil.copy(caller_script_path, destination_directory)

def str2bool(v):
    """Copy from TSLANet"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def test_classifier_head(classifer, classifer_name,
                    train_encoding, train_label, test_encoding, test_label,
                    base_path, setting, run_multiple=False):
    folder_path = os.path.join(base_path, setting)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    classifer_pipe = make_pipeline(
        StandardScaler(with_mean=True), 
        #RidgeClassifierCV(alphas=np.logspace(-3,3,10))
        classifer
    )
    if run_multiple:
        acc_all, bacc_all = [], []
        for train_enc, test_enc in zip(train_encoding, test_encoding):
            classifer_pipe.fit(train_enc, train_label)
            y_hat = classifer_pipe.predict(test_enc)
            acc_score__, bacc_score__ = accuracy_score(test_label, y_hat), balanced_accuracy_score(test_label, y_hat)
            acc_all.append(acc_score__)
            bacc_all.append(bacc_score__)
        acc_score, bacc_score = np.mean(acc_all), np.mean(bacc_all)
        if len(acc_all) >= 2:
            acc_std, bacc_std = np.std(acc_all), np.std(bacc_all)
            print(f"{classifer_name} accuracy: {acc_score:.4f} +/- {acc_std:.4f}")
            print(f"{classifer_name} b acc {bacc_score:.4f} +/- {bacc_std:.4f}")
        else:
            print(f"{classifer_name} accuracy: {acc_score:.4f}")
            print(f"{classifer_name} b acc {bacc_score:.4f}")
    
    else:
        classifer_pipe.fit(train_encoding, train_label)
        y_hat = classifer_pipe.predict(test_encoding)
        acc_score, bacc_score = accuracy_score(test_label, y_hat), balanced_accuracy_score(test_label, y_hat)

        print(f"{classifer_name} accuracy: {acc_score:.4f}")
        print(f"{classifer_name} b acc {bacc_score:.4f}")

    file_name='classification.txt'
    f = open(os.path.join(folder_path,file_name), 'a')
    f.write(setting + classifer_name + "  \n")
    f.write(f'accuracy:{acc_score:.5f}, balanced accuracy:{bacc_score:.5f}')
    f.write('\n')
    f.write('\n')
    f.close()


def test_latent(train_encoding, train_label, test_encoding, test_label, 
                  base_path, setting):
    folder_path = os.path.join(base_path, setting)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    knn = KNeighborsClassifier(5)
    knn.fit(train_encoding, train_label)
    pred = knn.predict(test_encoding)
    print(f"KNN accuracy: {accuracy_score(test_label, pred)}")
    print(f"KNN b acc {balanced_accuracy_score(test_label, pred)}")

    file_name='knn_classification.txt'
    f = open(os.path.join(folder_path,file_name), 'a')
    f.write(setting + "  \n")
    f.write(f'accuracy:{accuracy_score(test_label, pred)}, balanced accuracy:{balanced_accuracy_score(test_label, pred)}')
    f.write('\n')
    f.write('\n')
    f.close()


    all_encoding = np.concatenate([train_encoding, test_encoding], axis=0)
    all_label = np.concatenate([train_label, test_label], axis=0)

    if all_encoding.shape[0] > 50000:
        tsne_set_size = 50000
    else:
        tsne_set_size = all_encoding.shape[0]

    tsne_idx = np.random.choice(all_encoding.shape[0], size=tsne_set_size, replace=False)
    tsne_idx_train = tsne_idx[tsne_idx < train_encoding.shape[0]]
    tsne_idx_test = tsne_idx[tsne_idx >= train_encoding.shape[0]]

    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(all_encoding)

    # perform tSNE visualizer training for train data
    print(f"train set shape: ({len(tsne_idx_train)}, {train_encoding.shape[1]})")

    X_tsne_train = X_tsne[tsne_idx_train, :]
    train_label_tsne = all_label[tsne_idx_train]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne_train[:, 0], X_tsne_train[:, 1], c=train_label_tsne, cmap='Set2', alpha=0.5)
    plt.title('t-SNE plot test dataset')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')

    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)

    plt.savefig(os.path.join(folder_path, "tsne_train.png"))


    # perform tSNE visualizer testing for test data
    print(f"test set shape: ({len(tsne_idx_test)}, {test_encoding.shape[1]})")
    
    X_tsne_test = X_tsne[tsne_idx_test, :]
    test_label_tsne = all_label[tsne_idx_test]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne_test[:, 0], X_tsne_test[:, 1], c=test_label_tsne, cmap='Set2', alpha=0.5)
    plt.title('t-SNE plot test dataset')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')

    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)

    plt.savefig(os.path.join(folder_path, "tsne_test.png"))
    plt.close()