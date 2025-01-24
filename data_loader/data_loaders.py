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


# import numpy as np
# import torch
# from collections import Counter

# def oversample_eeg_dataset(eeg_data, labels, oversample_all=True, class_to_oversample=None, ratio=1.0):
#     """
#     Fungsi untuk melakukan oversampling pada dataset EEG.
    
#     Parameters:
#     ----------
#     eeg_data : torch.Tensor
#         Dataset EEG, dimensi (N, 3000, 1), dengan N segmen dan 3000 titik data per segmen.
    
#     labels : torch.Tensor
#         Label untuk dataset EEG, dimensi (N,), yang menunjukkan kelas untuk setiap segmen (5 kelas).
    
#     oversample_all : bool, default=True
#         Jika True, oversampling diterapkan pada semua kelas minoritas hingga jumlahnya sama dengan kelas mayoritas.
#         Jika False, oversampling diterapkan hanya pada kelas minoritas terkecil atau kelas yang dipilih.
    
#     class_to_oversample : int or None, default=None
#         Jika diberikan, hanya kelas ini yang akan dioversample sesuai dengan rasio. Jika None, kelas terkecil yang akan dioversample.
    
#     ratio : float, default=1.0
#         Rasio oversampling untuk kelas yang dipilih atau kelas minoritas terkecil.
    
#     Returns:
#     -------
#     torch.Tensor
#         Dataset EEG yang telah dioversampling (X_resampled).
#     torch.Tensor
#         Dataset label yang telah dioversampling (y_resampled).
#     """
#     # Menghitung jumlah segmen untuk masing-masing kelas
#     unique_classes, class_counts = torch.unique(labels, return_counts=True)
#     class_distribution = dict(zip(unique_classes.tolist(), class_counts.tolist()))
    
#     # Menentukan kelas mayoritas dan minoritas
#     majority_class = max(class_distribution, key=class_distribution.get)
#     minority_classes = {k: v for k, v in class_distribution.items() if v < class_distribution[majority_class]}
    
#     # Menentukan jumlah segmen yang perlu ditambahkan untuk kelas minoritas
#     if oversample_all:
#         # Oversample semua kelas minoritas hingga jumlahnya sama dengan kelas mayoritas
#         num_new_segments = {cls: class_distribution[majority_class] - count for cls, count in minority_classes.items()}
#     else:
#         # Jika oversample_all=False, pilih kelas terkecil atau kelas yang ditentukan
#         if class_to_oversample is None:
#             # Jika tidak ada kelas yang dipilih, pilih kelas dengan jumlah segmen terkecil
#             class_to_oversample = min(minority_classes, key=minority_classes.get)
        
#         # Menentukan jumlah segmen tambahan sesuai dengan rasio
#         num_new_segments = {class_to_oversample: int((class_distribution[majority_class] - class_distribution[class_to_oversample]) // ratio)}
    
#     # Menyiapkan tensor untuk data sintetis
#     synthetic_data = []
    
#     # Menghasilkan segmen sintetis untuk kelas minoritas
#     for class_id, num_segments in num_new_segments.items():
#         # Menyaring segmen-segmen kelas minoritas yang akan dioversample
#         minority_indices = (labels == class_id).nonzero(as_tuple=True)[0]
        
#         for _ in range(num_segments):
#             # Pilih segmen kelas minoritas secara acak
#             rand_idx = np.random.choice(minority_indices.numpy())
#             segment = eeg_data[rand_idx].numpy().flatten()  # Ambil segmen dan ubah menjadi vektor 1D
            
#             # Interpolasi antar titik data dalam segmen, tetapi menjaga jumlah titik tetap 3000
#             synthetic_segment = []
#             for i in range(len(segment) - 1):
#                 # Interpolasi linier antar titik data (menghasilkan nilai float)
#                 new_point = (segment[i] + segment[i + 1]) / 2.0
#                 synthetic_segment.append(segment[i])
#                 synthetic_segment.append(new_point)  # Menambah titik baru di tengah-tengah
            
#             # Tambahkan titik data terakhir
#             synthetic_segment.append(segment[-1])
            
#             # Menambahkan titik data sintetis ke dalam list
#             synthetic_data.append(synthetic_segment[:3000])  # Pastikan hanya mengambil 3000 titik
        
#     # Konversi data sintetis menjadi tensor (float)
#     synthetic_data_tensor = torch.tensor(synthetic_data, dtype=torch.float32).unsqueeze(-1)  # Ubah ke dimensi (num_synthetic_segments, 3000, 1)
    
#     # Gabungkan data asli dan data sintetis
#     combined_data = torch.cat([eeg_data, synthetic_data_tensor], dim=0)
    
#     # Label untuk data sintetis, pastikan hanya menambahkan label minoritas yang sesuai
#     if oversample_all:
#         combined_labels = torch.cat([labels, torch.cat([torch.full((synthetic_data_tensor.shape[0] // len(minority_classes),), cls) for cls in minority_classes])], dim=0)
#     else:
#         combined_labels = torch.cat([labels, torch.full((synthetic_data_tensor.shape[0],), class_to_oversample)], dim=0)
    
#     return combined_data, combined_labels


# # Fungsi oversampling yang telah dibuat sebelumnya
# def apply_oversampling(X_train, y_train, oversample_all=True, class_to_oversample=None, ratio=1.0):
#     """
#     Fungsi untuk menerapkan oversampling pada dataset X_train dan y_train.
    
#     Parameters:
#     ----------
#     X_train : numpy.ndarray
#         Fitur dataset training (N, 3000, 1).
    
#     y_train : numpy.ndarray
#         Label dataset training (N,).
    
#     oversample_all : bool, default=True
#         Jika True, oversampling diterapkan pada semua kelas minoritas hingga jumlahnya sama dengan kelas mayoritas.
#         Jika False, oversampling diterapkan hanya pada kelas minoritas terkecil atau kelas yang dipilih.
    
#     class_to_oversample : int or None, default=None
#         Jika diberikan, hanya kelas ini yang akan dioversample sesuai dengan rasio. Jika None, kelas terkecil yang akan dioversample.
    
#     ratio : float, default=1.0
#         Rasio oversampling untuk kelas yang dipilih atau kelas minoritas terkecil.
    
#     Returns:
#     -------
#     tuple: (numpy.ndarray, numpy.ndarray)
#         Dataset X yang telah dioversampling dan label y yang telah dioversampling.
#     """
#     # Menampilkan distribusi kelas sebelum oversampling
#     print(f"Distribusi kelas sebelum oversampling: {Counter(y_train)}")
    
#     # Mengubah X_train menjadi tensor jika masih berupa numpy array
#     if isinstance(X_train, np.ndarray):
#         X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     else:
#         X_train_tensor = X_train
        
#     if isinstance(y_train, np.ndarray):
#         y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#     else:
#         y_train_tensor = y_train
    
#     # Terapkan oversampling dengan fungsi yang telah dibuat
#     X_resampled, y_resampled = oversample_eeg_dataset(X_train_tensor, y_train_tensor, oversample_all, class_to_oversample, ratio)
    
#     # Menampilkan distribusi kelas setelah oversampling
#     if isinstance(y_resampled, torch.Tensor):
#         y_resampled = y_resampled.numpy()  # Convert to numpy array if it is still a tensor
#     print(f"Distribusi kelas setelah oversampling: {Counter(y_resampled)}")
    
#     # Mengembalikan hasil dalam format numpy array
#     return X_resampled.numpy(), y_resampled




# def data_generator_np(training_files, subject_files, batch_size, oversample_all=None, class_to_oversample=None, ratio=1.0):
#     """
#     Fungsi untuk memuat dataset, melakukan oversampling, dan mengembalikan DataLoader untuk training dan testing.
    
#     Parameters:
#     ----------
#     training_files : list of str
#         Daftar file training yang berisi dataset fitur dan label.
    
#     subject_files : list of str
#         Daftar file untuk dataset testing.
    
#     batch_size : int
#         Ukuran batch untuk DataLoader.
    
#     oversample_all : bool, default=True
#         Jika True, oversampling diterapkan pada semua kelas minoritas hingga jumlahnya sama dengan kelas mayoritas.
#         Jika False, oversampling diterapkan hanya pada kelas minoritas terkecil atau kelas yang dipilih.
    
#     class_to_oversample : int or None, default=None
#         Jika diberikan, hanya kelas ini yang akan dioversample sesuai dengan rasio. Jika None, kelas terkecil yang akan dioversample.
    
#     ratio : float, default=1.0
#         Rasio oversampling untuk kelas yang dipilih atau kelas minoritas terkecil.
    
#     Returns:
#     -------
#     train_loader : DataLoader
#         DataLoader untuk dataset training.
    
#     test_loader : DataLoader
#         DataLoader untuk dataset testing.
    
#     data_count : list
#         Jumlah segmen per kelas setelah oversampling.
#     """
#     # Memuat data training
#     X_train = np.load(training_files[0])["x"]
#     y_train = np.load(training_files[0])["y"]

#     # Menggabungkan data training dari beberapa file
#     for np_file in training_files[1:]:
#         X_train = np.vstack((X_train, np.load(np_file)["x"]))
#         y_train = np.append(y_train, np.load(np_file)["y"])

#     # Terapkan oversampling menggunakan apply_oversampling
#     X_resampled, y_resampled = apply_oversampling(X_train, y_train, oversample_all, class_to_oversample, ratio)

#     # Menampilkan distribusi kelas setelah oversampling
#     unique, counts = np.unique(y_resampled, return_counts=True)
#     data_count = list(counts)

#     # Membuat dataset dan dataloader untuk training
#     train_dataset = LoadDataset_from_numpy(X_resampled, y_resampled)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                drop_last=False,
#                                                num_workers=0)

#     # Memuat data testing
#     X_test = []
#     y_test = []
#     for np_file in subject_files:
#         data = np.load(np_file)
#         X_test.append(data["x"])
#         y_test.append(data["y"])
    
#     X_test = np.vstack(X_test)
#     y_test = np.concatenate(y_test)

#     # Membuat dataset dan dataloader untuk testing
#     test_dataset = LoadDataset_from_numpy(X_test, y_test)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                               batch_size=batch_size,
#                                               shuffle=False,
#                                               drop_last=False,
#                                               num_workers=0)

#     print(f"Distribusi Kelas Testing: {Counter(y_test)}")

#     return train_loader, test_loader, data_count


from imblearn.over_sampling import ADASYN
from collections import Counter

def apply_adasyn_1_1(X_train, y_train):
    class_counts = Counter(y_train)
    print(f"Distribusi kelas sebelum ADASYN: {class_counts}")
    
    class_2_count = class_counts[2]
    sampling_strategy = {1: class_2_count} 
    
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)  
    
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

    X_resampled, y_resampled = adasyn.fit_resample(X_train_reshaped, y_train)
    
    X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    print(f"Distribusi kelas setelah ADASYN: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def data_generator_np(training_files, subject_files, batch_size):
    X_train = np.load(training_files[0])["x"]
    y_train = np.load(training_files[0])["y"]

    for np_file in training_files[1:]:
        X_train = np.vstack((X_train, np.load(np_file)["x"]))
        y_train = np.append(y_train, np.load(np_file)["y"])

    X_resampled, y_resampled = apply_adasyn_1_1(X_train, y_train)

    unique, counts = np.unique(y_resampled, return_counts=True)
    data_count = list(counts)

    train_dataset = LoadDataset_from_numpy(X_resampled, y_resampled)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    X_test = []
    y_test = []
    for np_file in subject_files:
        data = np.load(np_file)
        X_test.append(data["x"])
        y_test.append(data["y"])
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    test_dataset = LoadDataset_from_numpy(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    print(f"Distribusi Kelas Testing: {Counter(y_test)}")

    return train_loader, test_loader, data_count

