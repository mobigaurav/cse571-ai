import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
         # Balancing the dataset
        self.balance_dataset()

    def balance_dataset(self):
        # Assuming the last column is the label
        collision_indices = np.where(self.normalized_data[:, -1] == 1)[0]
        no_collision_indices = np.where(self.normalized_data[:, -1] == 0)[0]

        # Random undersampling of the majority class
        no_collision_indices = np.random.choice(no_collision_indices, size=len(collision_indices), replace=False)
        balanced_indices = np.concatenate([collision_indices, no_collision_indices])

        # Update the dataset with balanced data
        self.normalized_data = self.normalized_data[balanced_indices]

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        sample = self.normalized_data[idx]
        x = sample[:-1].astype(np.float32)  # All columns except the last one
        y = np.array([sample[-1]]).astype(np.float32)  # Only the last column
        return {'input': x, 'label': y}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
       # Splitting dataset into train and test sets
        train_set, test_set = train_test_split(self.nav_dataset, test_size=0.2, random_state=42)
        # Creating DataLoaders
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
