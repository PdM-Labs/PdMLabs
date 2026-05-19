import torch.utils.data as data_utils
import torch
import pandas as pd
import numpy as np
from pdmlabs.method.USAD.usad import UsadModel,training,testing
from pdmlabs.method.USAD.utils import get_default_device,to_device
class usadCore():
    def __init__(self, window_size=10, num_epochs=15, lr=0.001,BATCH_SIZE=4,hidden_size=16,train_val=0.8,minmax=True, *args,
                 **kwargs):
        self.window_size = window_size
        self.BATCH_SIZE = BATCH_SIZE
        self.hidden_size = hidden_size
        self.N_EPOCHS = num_epochs
        self.lr=lr
        self.train_val=train_val
        self.device = get_default_device()
        self.model=None
        self.minmax=minmax
        self.forNorm={}
    def fit(self, df: pd.DataFrame) -> None:
        historic_data=df.copy()
        if self.minmax:
            for col in historic_data.columns:
                self.forNorm[col]=(historic_data[col].min(),historic_data[col].max())
                if historic_data[col].min()!=historic_data[col].max():
                    historic_data[col]=(historic_data[col] - self.forNorm[col][0] ) / (self.forNorm[col][1] - self.forNorm[col][0])
                else:
                    self.forNorm[col] = (historic_data[col].min(), historic_data[col].min())
                    historic_data[col]=[float(0) for i in range(len(historic_data.index))]

        while historic_data.shape[0] < self.window_size + 1 + self.BATCH_SIZE:
            point_to_pre_append = historic_data.iloc[0]
            historic_data = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=historic_data.columns),
                historic_data
            ])

        windows_normal = historic_data.values[np.arange(self.window_size)[None, :] + np.arange(historic_data.shape[0] - self.window_size)[:, None]]

        self.w_size = windows_normal.shape[1] * windows_normal.shape[2]
        z_size = windows_normal.shape[1] * self.hidden_size
        self.model = UsadModel(self.w_size, z_size)
        self.model = to_device(self.model, self.device)

        windows_normal_train = windows_normal[:int(np.floor(self.train_val* windows_normal.shape[0]))]
        windows_normal_val = windows_normal[
                             int(np.floor(self.train_val * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], self.w_size]))
        ), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0)

        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], self.w_size]))
        ), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0)

        history = training(self.N_EPOCHS, self.model, train_loader, val_loader,lr=self.lr)

    def predict(self, df: pd.DataFrame) -> list[float]:

        if self.model is None:
            assert False,"No trained USAD model for this source"
        target_data=df.copy()

        while target_data.shape[0] < self.window_size + self.BATCH_SIZE + 1:
            target_data = pd.concat([
                pd.DataFrame([target_data.iloc[0]], columns=target_data.columns),
                target_data
            ])

        if self.minmax:
            for col in target_data.columns:
                if self.forNorm[col][1]== self.forNorm[col][0]:
                    newcoldata=[0 if td==self.forNorm[col][0] else 1 for td in target_data[col].values]
                    target_data[col]=newcoldata
                else:
                    newcol=[]
                    for td in target_data[col].values:
                        diffminmax=self.forNorm[col][1]-self.forNorm[col][0]
                        if td < -2*diffminmax:
                            newcol.append(-1)
                        elif td >2*diffminmax:
                            newcol.append(2)
                        else:
                            newcol.append((td-self.forNorm[col][0])/diffminmax)
                    target_data[col]=newcol

        windows_attack = target_data.values[
            np.arange(self.window_size)[None, :] + np.arange(target_data.shape[0] - self.window_size)[:, None]]

        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0], self.w_size]))
        ), batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0)
        results = testing(self.model, test_loader)
        y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                 results[-1].flatten().detach().cpu().numpy()])
        y_pred=[sc for sc in y_pred]

        if len(y_pred) < len(df):
            diff=len(df)-len(y_pred)
            predictions=[y_pred[0] for i in range(diff)]
            predictions.extend(y_pred)
            return predictions
        elif len(y_pred) > len(df):
            return y_pred[-len(df):]

        return y_pred
    # TODO: predict one
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        pass