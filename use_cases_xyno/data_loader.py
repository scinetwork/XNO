from utils import MatReader
from torch.utils.data import DataLoader, TensorDataset, Dataset, default_collate

# Define the custom Dataset
class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


def _shape_printer(msg, x, y, xtes, ytes):
    print(f"\n=== {msg} ===\n")
    print(f"X_Train Shape: {x.shape}")
    print(f"Y_Train Shape: {y.shape}")
    print(f"X_Test Shape: {xtes.shape}")
    print(f"Y_Test Shape: {ytes.shape}")

def burger_1d(data_path, batch_size=16, resolution=50, ntrain=1000, ntest=100):
    
    ntrain = ntrain
    ntest = ntest
    sub = 2**3 #subsampling rate
    h = 2**13 // sub #total grid size divided by the subsampling rate
    s = h
    
    dataloader = MatReader(data_path)
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_train = x_train.reshape(ntrain,s,1)
    x_test = x_test.reshape(ntest,s,1)
    
    _shape_printer('Pure data structure', x_train, y_train, x_test, y_test)
    
    x_train = x_train.permute(0, 2, 1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.permute(0, 2, 1)
    y_test = y_test.unsqueeze(1)
    
    _shape_printer('Reshape data structure', x_train, y_train, x_test, y_test)

    train_loader = DictDataset(x_train, y_train)
    test_loader = DictDataset(x_test, y_test)
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
    test_loader = {
        resolution: test_loader
    }
    
    return train_loader, test_loader
