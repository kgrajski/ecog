import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, random_split, Dataset

from ecog import ECoGArrayRec

class ECoGDataSet(Dataset):
    """
    PyTorch custom Dataset tuned for ECoG Array Recordings.

    Args:
        Dataset (_type_): _description_
    """
    
    def __init__(self, study_dir, study_list_fname, label_list_fname, preproc_dir, transform=None, target_transform=None):
        self.samples = self.gen_dataset(study_dir, study_list_fname, label_list_fname, preproc_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples.path[idx], self.samples.label[idx]
        print("GET_ITEM=",idx,file_path,label)
        ecog_array = ECoGArrayRec().load(file_path)
        ecog_tensor = torch.tensor(ecog_array.ecog, dtype=torch.float32).unsqueeze(0) #unsqueeze to support batching later
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            ecog_tensor = self.transform(ecog_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        return ecog_tensor, label_tensor
    
    def gen_dataset(self, study_dir, study_list_fname, label_list_fname, preproc_dir):
        df_study = pd.read_csv(os.path.join(study_dir, study_list_fname), sep=',', header=0)
        df_labels = pd.read_csv(os.path.join(study_dir, label_list_fname), sep=',', header=0)
        df = []
        for row in df_study.itertuples():
            path = os.path.join(preproc_dir + os.sep + row.idkey + '.csv')
            df.append([row.idkey, path])
        df = pd.DataFrame(df,columns=['idkey','path'])
        df = pd.merge(df, df_labels, on='idkey', how='left').reset_index()
        return df[['path','label']]
