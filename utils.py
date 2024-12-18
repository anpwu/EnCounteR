import numpy as np
import pandas as pd

class myDataset():
    def __init__(self, df):
        self.X = df.filter(regex='^X').values
        self.U = df.filter(regex='^U').values
        self.T = df.filter(regex='^T').values
        self.W = self.T - self.T + 1
        self.Y = df.filter(regex='^Y').values

class loadIHDP:
    def __init__(self, args, path='./realdata/IHDP/', seed=2024, paramseed=2024) -> None:
        self.fmod = args.fmod
        self.tmod = args.tmod
        self.effect = args.effect
        self.rng = np.random.default_rng(seed)
        self.paramrng = np.random.default_rng(paramseed)
        
        train_data = np.load(path+'ihdp_npci_1-100.train.npz')
        test_data  = np.load(path+'ihdp_npci_1-100.test.npz')
        x_ = np.concatenate((train_data['x'][:,:,0],test_data['x'][:,:,0]),0)
        e_ = np.concatenate((train_data['t'][:,0],test_data['t'][:,0]),0)
        m0 = np.concatenate((train_data['mu0'][:,0:1],test_data['mu0'][:,0:1]),0)
        m1 = np.concatenate((train_data['mu1'][:,0:1],test_data['mu1'][:,0:1]),0)

        unique_counts = np.array([np.unique(x_[:, col]).size for col in range(6)])
        unique_column = [0,1,2,4,5]
        new_x = x_[:,unique_column]
        
        print("Unique Continuous Columns in IHDP: ", unique_counts)
        x = (new_x - np.mean(new_x, 0))/np.std(new_x, 0)
        u0 = (m0 - np.mean(m0, 0)) / np.std(m0, 0)
        u1 = (m1 - np.mean(m1, 0)) / np.std(m1, 0)

        data = np.concatenate([x,u0,u1],1)

        self.data0 = data[e_==0]
        self.data1 = data[e_==1]
        print('IHDP Encouragement Design (envs=2):', self.data0.shape, self.data1.shape)

        self.column = ['X' + str(i) for i in range(5)] + ['U5', 'U6', 'T', 'Y']
        self.set_params()

    def shuffle_data(self, data, val_num=75, test_num=75):
        self.rng.shuffle(data)
        train = data[test_num+val_num:]
        valid = data[test_num:test_num+val_num]
        test  = data[:test_num]

        return train, valid, test
    
    def backTE(self, X, U, alpha):
        alpha = alpha.reshape(-1,1)

        if self.fmod == 'Linear':
            return X @ alpha + U[:,0:1]
        else:
            return X @ alpha + X[:,1:] * X[:,:-1] @ alpha[:-1] + U[:,0:1]


    def backY(self, T, X, U, effect, beta):
        beta = beta.reshape(-1,1)

        if self.fmod == 'Linear':
            return T * effect + X @ beta + U[:,1:2]
        elif self.fmod == 'Mult.':
            Y = T * (X[:,0:1] + effect) + X @ beta
            Y = Y + X[:,1:] * X[:,:-1] @ beta[:-1]
            Y = Y + U[:,1:2]
            return  Y
        else:
            return T * effect + X @ beta + U[:,1:2]

    def generate_data(self):
        train1, valid, test = self.shuffle_data(self.data0)
        train2 = self.data1

        dfs = [self.get_env(train1,0), self.get_env(train2,1), self.get_env(valid,0)]
        datas = [myDataset(df) for df in dfs]

        df_sorted, test_data = self.get_test(test)

        return dfs, datas, df_sorted, test_data

    def get_test(self, proxy):
        expanded_proxy = np.tile(proxy, (100, 1))
        X, U = expanded_proxy[:,:5], expanded_proxy[:,5:]
        T = self.rng.uniform(0.0, 1.0, size=(75*100, 1)).round(1)
        YT = self.backY(T, X, U, self.effect, self.beta)
        Y0 = self.backY(T-T, X, U, self.effect, self.beta)
        Y1 = self.backY(T-T+1, X, U, self.effect, self.beta)
        df = pd.DataFrame(np.concatenate([X, U, T, YT, Y0, Y1], axis=1), columns=self.column + ['Y0', 'Y1'])
        df_sorted = df.sort_values(by='T')
        test_data = myDataset(df_sorted)
        return df_sorted, test_data

        
    def get_env(self, proxy, env):
        X, U = proxy[:,:5], proxy[:,5:]
        T = self.backTE(X, U, self.alpha[:,env])
        T = np.abs(T)
        Y = self.backY(T, X, U, self.effect, self.beta)
        df = pd.DataFrame(np.concatenate([X, U, T, Y], axis=1), columns=self.column)
        return df
    
    def set_params(self, alpha=None, beta=None):
        self.set_alpha(alpha)
        self.set_beta(beta)

    def set_alpha(self, alpha=None):
        if alpha is None:
            self.alpha = self.paramrng.uniform(0.1, 0.9, size=(5, 3)).round(1)
        else:
            self.alpha = alpha
    
    def set_beta(self, beta=None):
        if beta is None:
            self.beta = self.paramrng.uniform(0.1, 0.9, size=(5, 1)).round(1)
        else:
            self.beta = beta

    def df2data(self, df):
        data = myDataset(df)
        return data