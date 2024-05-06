import torch
import time
import math
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from matplotlib import pyplot
from torch import cosine_similarity
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from Ob_propagation import Observation_progation

lr = 0.1
d_inp = 20
d_model = 144
d_static = 0
nhid = 110
nlayers = 5
nhead =2
dropout = 0
max_len = 300
split=0.6
input_window = 10
output_window = 2
batch_size = 36
epochs = 200
Normalization = True#False#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sensor_wise_mask = False
feature_size = d_model+16
global_structure = torch.ones(d_inp, d_inp)

aggreg = 'max'
start = time.perf_counter()
print("start:", start)
torch.manual_seed(0)
np.random.seed(0)
calculate_loss_over_all_values = False


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window], output_window * [0])
        train_label = input_data[i:i+tw]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data():

    from pandas import read_csv
    series = read_csv('cubic.csv', header=0, index_col=0, parse_dates=False, squeeze=True)
    if Normalization == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
        amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    else:
        amplitude = (series.to_numpy().reshape(-1, 1)).reshape(-1)
    global df_amp
    df_amp = pd.DataFrame(amplitude)
    train_data = amplitude[:int(len(amplitude) * split)]
    test_data = amplitude[int(len(amplitude) * split):len(amplitude)]
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?
    return train_sequence.to(device),test_data.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def generate_global_structure(data, K=3):
    observations = data[:, :, :18]
    cos_sim = torch.zeros([observations.shape[0], 18, 18])
    for row in tqdm(range(observations.shape[0])):
        unit = observations[row].T
        cos_sim_unit = cosine_similarity(unit)
        cos_sim[row] = torch.from_numpy(cos_sim_unit)
    ave_sim = torch.mean(cos_sim, dim=0)
    index = torch.argsort(ave_sim, dim=0)
    index_K = index < K
    global_structure = index_K * ave_sim
    global_structure = F.sigmoid(global_structure)
    return global_structure


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 300):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class GNN_TS(nn.Module):


    def __init__(self, d_inp=0, d_model=0, nhead=0, nhid=0, nlayers=0, dropout=0, max_len=0,
                      perc=0.5, aggreg='max', global_structure=None, sensor_wise_mask=True,
                     static=False):


        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask
        d_pe = 15
        d_enc = d_inp
        self.d_inp = d_inp
        self.d_model = d_model
        self.d_ob = int(d_model/d_inp)
        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.adj = torch.ones([self.d_inp, self.d_inp])
        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob))

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)
        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.Sigmoid(),
            nn.Linear(d_final, 1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
            nn.Linear(d_model, 1),
        )
        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.38
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src):
        self.src_mask = None
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def train(train_data):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.22)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
    # test_result_np = test_result.numpy()
    # truth_np = truth.numpy()
    # df = pd.DataFrame(test_result_np)
    # df1 = pd.DataFrame(truth_np)
    # df.to_csv('test_results.csv', index=False)
    # df1.to_csv('truth.csv', index=False)


    pyplot.plot(test_result,color="red")
    pyplot.plot(truth, color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.ylim(0, 1.6)
    pyplot.savefig('graph/validation_epoch%d.png'%epoch)
    pyplot.close()

    return total_loss / i

def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    _, data = get_batch(data_source, len(data_source) - 2, len(data_source) - 1)
    with torch.no_grad():
        for i in range(0, steps, 1):

            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))
    data = data.cpu().view(-1)
    pyplot.plot(data,color="red")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/predict%d.png')
    pyplot.close()


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 10
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()

    return total_loss / len(data_source)


model = GNN_TS(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                     0.5, aggreg, global_structure,
                                    sensor_wise_mask=sensor_wise_mask).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler: StepLR = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.98)
best_val_loss = float("inf")
best_model = None


if __name__ == "__main__":
    train_data, test_data = get_data()
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        if(epoch % 10 == 0):
            val_loss = plot_and_loss(model, test_data, epoch)
            predict_future(model, test_data, 200)###预测数据的长度
        else:
            val_loss = evaluate(model, test_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        scheduler.step()
    end = time.perf_counter()
    print("\nend:", end)
    duration = end - start
    print("\nThe duration is:", duration)
