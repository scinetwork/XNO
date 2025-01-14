from utils import MatReader
from torch.utils.data import DataLoader, TensorDataset, Dataset, default_collate
from xno.data.datasets import load_darcy_flow_small
from xno.data.datasets import load_navier_stokes_pt
from xno.data.transforms.normalizers import UnitGaussianNormalizer

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

def _1d_burger(data_path, batch_size=16, resolution=50, ntrain=1000, ntest=100):
    
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

def _2d_darcy(data_path, batch_size=16, resolution=32, ntrain=200, ntest=100):
    train_loader, test_loader, output_encoder = load_darcy_flow_small(
    n_train=ntrain,
    batch_size=batch_size,
    test_resolutions=[resolution],
    n_tests=[ntest],
    test_batch_sizes=[batch_size],
    )
    
    return train_loader, test_loader

def _2d_navier_stoke(data_path, batch_size=16, resolution=128, ntrain=200, ntest=100):
    train_loader, test_loader, output_encoder = load_navier_stokes_pt(
    n_train=ntrain,
    batch_size=batch_size,
    test_resolutions=[resolution],
    n_tests=[ntest],
    test_batch_sizes=[batch_size],
    )
    
    return train_loader, test_loader

def _3d_navier_stoke(data_path, batch_size=16, resolution=64, ntrain=1000, ntest=200, regularization=False):
    
    sub = 1
    S = 64 // sub
    T_in = 10
    T = 10
    
    reader = MatReader(data_path)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

    reader = MatReader(data_path)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

    assert (S == train_u.shape[-2])
    assert (T == train_u.shape[-1])

    if regularization:
        a_normalizer = UnitGaussianNormalizer(dim=[0, 1, 2, 3])
        a_normalizer.fit(train_a)
        train_a = a_normalizer.transform(train_a)
        test_a = a_normalizer.transform(test_a)

        y_normalizer = UnitGaussianNormalizer(dim=[0, 1, 2, 3]) 
        y_normalizer.fit(train_u)  
        train_u = y_normalizer.transform(train_u) 

    train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
    test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

    _shape_printer('Pure data structure', train_a, train_u, test_a, test_u)
    
    x_train = train_a.permute(0, 4, 3, 2, 1)
    y_train = train_u.unsqueeze(1)
    y_train = y_train.permute(0, 1, 4, 3, 2)
    x_test = test_a.permute(0, 4, 3, 2, 1)
    y_test = test_u.unsqueeze(1)
    y_test = y_test.permute(0, 1, 4, 3, 2)
    
    _shape_printer('Reshape data structure', x_train, y_train, x_test, y_test)
    
    train_loader = DictDataset(x_train, y_train)
    test_loader = DictDataset(x_test, y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
    test_loader = {
        resolution: test_loader
    }
    
    return train_loader, test_loader