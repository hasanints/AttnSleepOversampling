import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.combine import SMOTEENN


class LoadDataset_from_numpy(Dataset):
    def __init__(self, X_data, y_data):
        super(LoadDataset_from_numpy, self).__init__()
        self.x_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()

        # Reshape to (Batch_size, #channels, seq_len)
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# Fungsi oversampling yang telah dibuat sebelumnya
def apply_oversampling(X_train, y_train, oversample_all=True, class_to_oversample=None, ratio=1.0):
    """
    Fungsi untuk menerapkan oversampling pada dataset X_train dan y_train.
    
    Parameters:
    ----------
    X_train : numpy.ndarray
        Fitur dataset training (N, 3000, 1).
    
    y_train : numpy.ndarray
        Label dataset training (N,).
    
    oversample_all : bool, default=True
        Jika True, oversampling diterapkan pada semua kelas minoritas hingga jumlahnya sama dengan kelas mayoritas.
        Jika False, oversampling diterapkan hanya pada kelas minoritas terkecil atau kelas yang dipilih.
    
    class_to_oversample : int or None, default=None
        Jika diberikan, hanya kelas ini yang akan dioversample sesuai dengan rasio. Jika None, kelas terkecil yang akan dioversample.
    
    ratio : float, default=1.0
        Rasio oversampling untuk kelas yang dipilih atau kelas minoritas terkecil.
    
    Returns:
    -------
    tuple: (numpy.ndarray, numpy.ndarray)
        Dataset X yang telah dioversampling dan label y yang telah dioversampling.
    """
    # Menampilkan jumlah kelas sebelum oversampling
    print(f"Distribusi kelas sebelum oversampling: {Counter(y_train)}")
    
    # Mengubah X_train menjadi tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # Terapkan oversampling dengan fungsi yang telah dibuat
    X_resampled, y_resampled = oversample_eeg_dataset(X_train_tensor, y_train_tensor, oversample_all, class_to_oversample, ratio)
    
    # Menampilkan jumlah kelas setelah oversampling
    print(f"Distribusi kelas setelah oversampling: {Counter(y_resampled)}")
    
    # Mengembalikan hasil dalam format numpy array
    return X_resampled.numpy(), y_resampled.numpy()


def data_generator_np(training_files, subject_files, batch_size, oversample_all=True, class_to_oversample=None, ratio=1.0):
    """
    Fungsi untuk memuat dataset, melakukan oversampling, dan mengembalikan DataLoader untuk training dan testing.
    
    Parameters:
    ----------
    training_files : list of str
        Daftar file training yang berisi dataset fitur dan label.
    
    subject_files : list of str
        Daftar file untuk dataset testing.
    
    batch_size : int
        Ukuran batch untuk DataLoader.
    
    oversample_all : bool, default=True
        Jika True, oversampling diterapkan pada semua kelas minoritas hingga jumlahnya sama dengan kelas mayoritas.
        Jika False, oversampling diterapkan hanya pada kelas minoritas terkecil atau kelas yang dipilih.
    
    class_to_oversample : int or None, default=None
        Jika diberikan, hanya kelas ini yang akan dioversample sesuai dengan rasio. Jika None, kelas terkecil yang akan dioversample.
    
    ratio : float, default=1.0
        Rasio oversampling untuk kelas yang dipilih atau kelas minoritas terkecil.
    
    Returns:
    -------
    train_loader : DataLoader
        DataLoader untuk dataset training.
    
    test_loader : DataLoader
        DataLoader untuk dataset testing.
    
    data_count : list
        Jumlah segmen per kelas setelah oversampling.
    """
    # Memuat data training
    X_train = np.load(training_files[0])["x"]
    y_train = np.load(training_files[0])["y"]

    # Menggabungkan data training dari beberapa file
    for np_file in training_files[1:]:
        X_train = np.vstack((X_train, np.load(np_file)["x"]))
        y_train = np.append(y_train, np.load(np_file)["y"])

    # Terapkan oversampling menggunakan apply_oversampling
    X_resampled, y_resampled = apply_oversampling(X_train, y_train, oversample_all, class_to_oversample, ratio)

    # Menampilkan distribusi kelas setelah oversampling
    unique, counts = np.unique(y_resampled, return_counts=True)
    data_count = list(counts)

    # Membuat dataset dan dataloader untuk training
    train_dataset = LoadDataset_from_numpy(X_resampled, y_resampled)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    # Memuat data testing
    X_test = []
    y_test = []
    for np_file in subject_files:
        data = np.load(np_file)
        X_test.append(data["x"])
        y_test.append(data["y"])
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    # Membuat dataset dan dataloader untuk testing
    test_dataset = LoadDataset_from_numpy(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    print(f"Distribusi Kelas Testing: {Counter(y_test)}")

    return train_loader, test_loader, data_count


# from imblearn.over_sampling import ADASYN
# from collections import Counter

# def apply_adasyn_1_1(X_train, y_train):
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum ADASYN: {class_counts}")
    
#     class_2_count = class_counts[2]
#     sampling_strategy = {1: class_2_count} 
    
#     adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)  
    
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#     X_resampled, y_resampled = adasyn.fit_resample(X_train_reshaped, y_train)
    
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
#     print(f"Distribusi kelas setelah ADASYN: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled


# def data_generator_np(training_files, subject_files, batch_size):
#     X_train = np.load(training_files[0])["x"]
#     y_train = np.load(training_files[0])["y"]

#     for np_file in training_files[1:]:
#         X_train = np.vstack((X_train, np.load(np_file)["x"]))
#         y_train = np.append(y_train, np.load(np_file)["y"])

#     X_resampled, y_resampled = apply_adasyn_1_1(X_train, y_train)

#     unique, counts = np.unique(y_resampled, return_counts=True)
#     data_count = list(counts)

#     train_dataset = LoadDataset_from_numpy(X_resampled, y_resampled)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                drop_last=False,
#                                                num_workers=0)

#     X_test = []
#     y_test = []
#     for np_file in subject_files:
#         data = np.load(np_file)
#         X_test.append(data["x"])
#         y_test.append(data["y"])
    
#     X_test = np.vstack(X_test)
#     y_test = np.concatenate(y_test)

#     test_dataset = LoadDataset_from_numpy(X_test, y_test)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                               batch_size=batch_size,
#                                               shuffle=False,
#                                               drop_last=False,
#                                               num_workers=0)
#     print(f"Distribusi Kelas Testing: {Counter(y_test)}")

#     return train_loader, test_loader, data_count

