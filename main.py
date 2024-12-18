import os
import argparse
import numpy as np
import pandas as pd
import random

from utils import myDataset, loadIHDP
from src import torchDataset, MyNetwork, set_seed, RepNetwork
from src import WNetwork, wMCov, wMean, wSTD
import torch
import torch.optim as optim


def set_args():
    parser = argparse.ArgumentParser(description='Encouragement Design')
    parser.add_argument('--exps', type=int, default=10, help='Number of Repetition')
    parser.add_argument('--effect', type=float, default=0.50, help='Causal Effect')
    parser.add_argument('--seed', type=int, default=2024, help='Seed of Algorithm')
    parser.add_argument('--paramseed', type=int, default=2024, help='Seed of Cofficient')
    parser.add_argument('--envs', type=int, default=2, help='Number of environments')
    parser.add_argument('--split', type=str, default='[0,0]', help='Should Split')
    parser.add_argument('--num', type=str, default='[458,139]', help='Sample size of Encouragements')
    parser.add_argument('--valnum', type=int, default=75, help='Sample size of validation data')
    parser.add_argument('--testnum', type=int, default=7500, help='Sample size of testing data')
    ###################################################
    parser.add_argument('--diff', type=int, default=1, help='Imbalance Mode')
    parser.add_argument('--mean', type=str, default='[0.1,-0.1]', help='Mean of Imbalance Data')
    parser.add_argument('--std', type=str, default='[0.7,1.2]', help='STD of Imbalance Data')
    parser.add_argument('--rho', type=str, default='[0.6,0.4]', help='Imbalance Ratio')
    ###################################################
    parser.add_argument('--dim', type=int, default=7, help='Dimension of X and U')
    parser.add_argument('--xdim', type=int, default=5, help='Dimension of X')
    parser.add_argument('--rdim', type=int, default=15, help='[1,2,5,12,15,20]')
    parser.add_argument('--hdim', type=int, default=128, help='[16,32,64,128,257]')
    parser.add_argument('--alpha', type=int, default=5, help='[1,5,8,10,12,20]')
    parser.add_argument('--covx', type=float, default=0.3, help='Covariance')
    parser.add_argument('--covu', type=float, default=0.3, help='Covariance')
    ###################################################
    parser.add_argument('--fmod', type=str, default='Mult.', help='Linear/Mult./')
    parser.add_argument('--tmod', type=str, default='Abs', help='Abs')
    ###################################################
    parser.add_argument('--epoch3', type=int, default=1000, help='Full-Training-Epoch')
    parser.add_argument('--epoch2', type=int, default=100, help='Adversarial-Training-Epoch')
    parser.add_argument('--epoch1', type=int, default=10, help='Reweighting-Training-Epoch')
    parser.add_argument('--lrate', type=float, default=0.005, help='Learning Rate')
    parser.add_argument('--loss', type=str, default=None, help='Vanilla: reg')
    parser.add_argument('--mods', type=str, default='[1,0,1,1,1,1]', help='''mods[0]: balance,mods[1]: Weight,
                        mods[2]: MSE, mods[3]: Mas0 + mean&std, mods[4]: X, mods[5]: R. ''')
    ###################################################
    parser.add_argument('--ifprint', type=bool, default=True, help='Print or not')
    parser.add_argument('--dataPath', type=str, default='./datas/', help='Data Path')
    parser.add_argument('--resultPath', type=str, default='./results/', help='Result Path')
    args = parser.parse_args()
    return args

def moment_objective_reweight(data, W, net, rep):
    epsilon = data.Y - net(data.T, data.X)
    factor = rep(data.T, data.X)
    X_cent = data.X - torch.mean(data.X, 0)

    mean = wMean(epsilon, W)
    std  = wSTD(epsilon, W)
    moment_Eas0 = (mean) ** 2
    moment_X = torch.sum((wMean(epsilon * X_cent, W) * wSTD(X_cent, W)) ** 2)
    moment_R = torch.sum((wMean(epsilon * factor, W) * wSTD(factor, W)) ** 2)
    moment_MSE = wMean(epsilon**2, W)

    return mean, std, moment_Eas0, moment_X, moment_R, moment_MSE

def moment_objective_unweight(data, W, net, rep):
    epsilon = data.Y - net(data.T, data.X)
    factor = rep(data.T, data.X)
    X_cent = data.X - torch.mean(data.X, 0)

    mean = wMean(epsilon, W)
    std  = wSTD(epsilon, W)
    moment_Eas0 = mean ** 2
    moment_X = torch.sum(wMean(epsilon * X_cent, W) ** 2)
    moment_R = torch.sum(wMean(epsilon * factor, W) ** 2)
    moment_MSE = wMean(epsilon**2, W)

    return mean, std, moment_Eas0, moment_X, moment_R, moment_MSE

def balance_objective(datas, Wnets):
    full_Mean, full_Cov = wMCov(torch.cat([data.X for data in datas]))
    balance_MSE, Ws = 0, []
    for i in range(len(datas)):
        W, Mean, Cov = Wnets[i](datas[i].X)
        balance_MSE = balance_MSE + torch.sum((full_Mean - Mean) ** 2) + torch.sum((full_Cov - Cov) ** 2)
        Ws.append(W)

    return Ws, balance_MSE


def objectives(datas, Wnets, net, rep, mods=[0,1,1,1,1,1], loss=None):

    Ws, balance_MSE = balance_objective(datas, Wnets)
    if loss == 'balance':
        return balance_MSE
    if mods[0] == 0:
        for i in range(len(Ws)):
            Ws[i] = Ws[i] - Ws[i] + 1

    means, stds, moment_Eas0s, moment_Xs, moment_Rs, moment_MSEs = [], [], 0, 0, 0, 0
    for i in range(len(datas)):
        if mods[1] == 1:
            mean, std, moment_Eas0, moment_X, moment_R, moment_MSE = moment_objective_reweight(datas[i], Ws[i], net, rep)
        else:
            mean, std, moment_Eas0, moment_X, moment_R, moment_MSE = moment_objective_unweight(datas[i], Ws[i], net, rep)
        means.append(mean)
        stds.append(std)
        moment_Eas0s = moment_Eas0s + moment_Eas0
        moment_Xs = moment_Xs + moment_X
        moment_Rs = moment_Rs + moment_R
        moment_MSEs = moment_MSEs + moment_MSE
    if loss == 'reg':
        return moment_MSEs
    if loss == 'adversial':
        return -moment_Rs
    
    means, stds = torch.cat(means), torch.cat(stds)
    moment_mean = torch.sum((means.unsqueeze(0) - means.unsqueeze(1))**2) / 2
    moment_std  = torch.sum((stds.unsqueeze(0) - stds.unsqueeze(1))**2) / 2
    moment_mean_std = moment_mean + moment_std

    return mods[2] * moment_MSEs + mods[3] * (moment_Eas0s + moment_mean_std) + mods[4] * moment_Xs + mods[5] * moment_Rs


def test_eval(exp, data, eval, ifprint=True):
    early = np.mean(((eval.Y[:,0:1]-net(eval.T, eval.X)).detach().numpy())**2)

    hat_YT = net(data.T, data.X).detach().numpy()
    hat_Y0 = net(data.T-data.T, data.X).detach().numpy()
    hat_Y1 = net(data.T-data.T+1, data.X).detach().numpy()
    hat_CATE = hat_YT-hat_Y0
    hat_CATEv1 = hat_Y1-hat_Y0

    YT = data.Y[:,0:1].detach().numpy()
    Y0 = data.Y[:,1:2].detach().numpy()
    Y1 = data.Y[:,2:3].detach().numpy()
    CATE = YT-Y0
    CATEv1 = Y1-Y0
    
    YT_MSE, Y1_MSE, Y0_MSE, PEHE, PEHEv1 = np.mean((YT-hat_YT)**2), np.mean((Y1-hat_Y1)**2), np.mean((Y0-hat_Y0)**2), np.sqrt(np.mean((CATE-hat_CATE)**2)), np.sqrt(np.mean((CATEv1-hat_CATEv1)**2))

    ATE = np.mean(CATEv1)
    hat_ATE = np.mean(hat_CATEv1)
    epsilon_ATE = np.abs(ATE-hat_ATE)

    total_np = [[exp, early, YT_MSE, PEHE, epsilon_ATE, YT_MSE, Y1_MSE, Y0_MSE, ATE, hat_ATE, epsilon_ATE, np.mean(Y1), np.mean(hat_Y1), np.mean(Y0), np.mean(hat_Y0)]]
    columns = ['exp', 'eval', 'MSE', 'PEHE', 'ATE_bias', 'YT_MSE', 'Y1_MSE', 'Y0_MSE', 'ATE', 'hat_CATE', 'epsilon_ATE', 'Y1', 'hat_Y1', 'Y0', 'hat_Y0']
    total_df = pd.DataFrame(np.array(total_np), columns=columns).round(3)
    if ifprint: print(exp, early, YT_MSE, PEHE, epsilon_ATE)

    t_results = []
    for t_value in np.unique(data.T):
        t_results.append([np.sqrt(np.mean(((CATE-hat_CATE)[data.T==t_value])**2)),np.abs(np.mean((CATEv1-hat_CATEv1)[data.T==t_value]))])
    t_df = pd.DataFrame(np.array(t_results), columns=['PEHE', 'ATE_bias']).round(3).T
    t_df.columns = np.unique(data.T)
    
    return total_df, t_df, YT_MSE, Y1_MSE, Y0_MSE, PEHE, ATE, hat_CATE, epsilon_ATE, hat_CATE, CATE

args = set_args()
random.seed(2024)

if __name__ == "__main__":
    num_list = [f'{eval(args.num)[i]-eval(args.split)[i]}' for i in range(args.envs)]
    num_str = '['+ ','.join(num_list) + ']'
    data_name = f'{args.fmod}_{args.tmod}_{args.xdim}_{args.dim-args.xdim}_{num_str}'
    mods, loss = eval(args.mods), args.loss
    mods[3],mods[4],mods[5] = mods[3]*args.alpha,mods[4]*args.alpha,mods[5]*args.alpha
    modules_name = f'{args.mods}_{args.alpha}_{args.rdim}_{args.hdim}'
    print(modules_name, args.loss, args.seed, ':', data_name)


    Design = loadIHDP(args, './realdata/IHDP/', args.seed)
    if not os.path.exists(f'{args.dataPath}/{data_name}/{args.exps-1}/'):
        for exp in range(args.exps):
            print(f'Data Generation: {exp}.')
            dfs, datas, test_df, test_data = Design.generate_data()
            
            data_path = f'{args.dataPath}/{data_name}/{exp}/'
            os.makedirs(data_path, exist_ok=True)
            for i, df in enumerate(dfs):
                df.to_csv(data_path+f'data{i}.csv', index=False)
            test_df.to_csv(data_path+'test.csv', index=False)
            np.savetxt(data_path+'parameters.csv', np.concatenate([Design.alpha, Design.beta],1).T, fmt="%.2f", delimiter=",")

    # Setup Training
    result_dfs = pd.DataFrame()
    for exp in range(args.exps):
        print(f'Begin Trainning({exp}): {data_name} - {modules_name}.')
        os.makedirs(args.resultPath + f'/{data_name}/{modules_name}/Total/{exp}/', exist_ok=True)
        os.makedirs(args.resultPath + f'/{data_name}/{modules_name}/ATE/{exp}/', exist_ok=True)
        os.makedirs(args.resultPath + f'/{data_name}/{modules_name}/PEHE/{exp}/', exist_ok=True)

        set_seed(args.seed+exp)
        data_path = f'{args.dataPath}/{data_name}/{exp}/'
        dfs = [pd.read_csv(data_path+f'data{i}.csv', index_col=False) for i in range(args.envs+1)]
        datas = [myDataset(df) for df in dfs]
        test_data = myDataset(pd.read_csv(data_path+'test.csv', index_col=False))
        train_datas = [torchDataset(datas[i]) for i in range(args.envs)]
        valid_data = torchDataset(datas[-1])
        test_data = torchDataset(test_data)

        Wnets = [WNetwork(len(train_datas[i].X)) for i in range(args.envs)]
        net = MyNetwork(1, args.xdim, args.hdim)
        rep = RepNetwork(1, args.xdim, args.rdim, args.hdim)

        optim1 = optim.Adam(net.parameters(), lr=args.lrate)
        optim2 = optim.Adam(rep.parameters(), lr=args.lrate)
        Wnet_parameters = list()
        for Wnet in Wnets:
            Wnet_parameters = Wnet_parameters + list(Wnet.parameters())
        optim3 = optim.Adam(Wnet_parameters, lr=args.lrate)

        df_tot = pd.DataFrame()
        df_ate = pd.DataFrame()
        df_pehe = pd.DataFrame()


        # Begin Training
        for itr in range(args.epoch1):
            net.eval()
            rep.eval()
            for Wnet in Wnets:
                Wnet.train()

            loss3 = objectives(train_datas, Wnets, net, rep, mods, loss='balance')

            optim3.zero_grad()
            loss3.backward()
            optim3.step()


        for itr in range(args.epoch3+1):
            net.train()
            rep.eval()
            for Wnet in Wnets:
                Wnet.eval()
            
            loss1 = objectives(train_datas, Wnets, net, rep, mods, loss=loss)

            optim1.zero_grad()
            loss1.backward()
            optim1.step()

            if mods[5]>0:
                if itr < args.epoch2:
                    rep.train()
                    net.eval()
                    for Wnet in Wnets:
                        Wnet.eval()

                    loss2 = objectives(train_datas, Wnets, net, rep, mods, loss='adversial')

                    optim2.zero_grad()
                    loss2.backward()
                    optim2.step()
            
            net.eval()
            rep.eval()
            if itr % 100 == 0:
                total_df, t_df, YT_MSE, Y1_MSE, Y0_MSE, PEHE, ATE, hat_CATE, epsilon_ATE, _, _ = test_eval(itr, test_data, valid_data, args.ifprint)
                
                df_tot = pd.concat([df_tot, total_df])
                df_ate = pd.concat([df_ate, t_df.iloc[1:2]])
                df_pehe = pd.concat([df_pehe, t_df.iloc[0:1]])
                df_tot.to_csv(f'{args.resultPath}/{data_name}/{modules_name}/Total/{exp}/model_tot.csv')
                df_ate.to_csv(f'{args.resultPath}/{data_name}/{modules_name}/ATE/{exp}/model_ate.csv')
                df_pehe.to_csv(f'{args.resultPath}/{data_name}/{modules_name}/PEHE/{exp}/model_pehe.csv')

        result_dfs = pd.concat([result_dfs, df_tot[-1:]])
        result_dfs = result_dfs.reset_index(drop=True)
        result_dfs.to_csv(f'{args.resultPath}/{data_name}/{modules_name}/result.csv')
        
