import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Classe do modelo da Rede Neural
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.Sigmoid(output)
        return output
def redeNeural(nome, momentum, lr, epocas, hiddenSize, datasetNome):
    momentum = float(momentum)
    lr = float(lr)
    epocas = int(epocas)
    hiddenSize = int(hiddenSize)
    start = time.perf_counter()
    df = pd.read_csv('D1.csv')
    df = df.replace(np.nan, 0)
    df2 = df.iloc[18:]
    df2['FTSG'] = df2['FTHG'] + df2['FTAG']
    df2['Over1.5'] = [1 if x > 1.5 else 0 for x in df2["FTSG"]]
    colunas = ['HomeTeam', 'AwayTeam', 'Over1.5', 'EGHT', 'EGAT',
               'BPHT', 'BPAT', 'GAHT', 'GAAT', 'SoGHT', 'SoGAT', 'FKHT', 'FKAT',
               'CKHT', 'CKAT', 'OHT', 'OAT', 'THT', 'TAT', 'GSHT', 'GSAT', 'FHT',
               'FAT', 'RCHT', 'RCAT', 'YCHT', 'YCAT', 'TPHT', 'TPAT', 'AHT', 'AAT',
               'DAHT', 'DAAT', 'CCHT', 'CCAT']
    df2 = df2[colunas]
    media_times = df.iloc[9:18]  # Corrigir a seleção de dados
    colunas.remove('Over1.5')
    media_times = media_times[colunas]
    media_times = media_times.groupby(['HomeTeam', 'AwayTeam']).mean()
    media_times.reset_index(inplace=True)
    training = df2
    test = media_times

    training = training.sample(frac=1)
    test = test.sample(frac=1)

    nomes_training = df2[['HomeTeam', 'AwayTeam']]
    nomes_test = media_times[['HomeTeam', 'AwayTeam']]

    training_input = df2.iloc[:, 3:]
    training_output = df2['Over1.5']
    test_input = media_times.iloc[:, 2:]  # Corrigir a seleção de colunas
    test_output = df2['Over1.5']  # Corrigir a seleção de colunas

    # Normalizar os dados
    scaler = StandardScaler()
    training_input = scaler.fit_transform(training_input)
    test_input = scaler.transform(test_input)

    # Convertendo para tensor
    training_input = torch.FloatTensor(training_input)
    training_output = torch.FloatTensor(training_output.values)
    test_input = torch.FloatTensor(test_input)
    test_output = torch.FloatTensor(test_output.values)

    # Criar a instância do modelo
    input_size = training_input.size()[1]
    hidden_size = hiddenSize
    model = Net(input_size, hidden_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Treinamento
    model.train()
    epochs = epocas
    errors = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Fazer o forward
        y_pred = model(training_input)
        # Cálculo do erro
        loss = criterion(y_pred.squeeze(), training_output.squeeze())
        errors.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Época: {epoch} Loss: {loss.item()}')
        # Backpropagation
        loss.backward()
        optimizer.step()

    # Testar o modelo já treinado
    end = time.perf_counter()
    tempo_total = end - start
    model.eval()
    y_pred = model(test_input)
    erro_pos_treinamento = criterion(y_pred.view(-1), test_output[:len(y_pred)].view(-1))  # Ajustar o tamanho
    predicted = y_pred.detach().numpy()
    real = test_output[:len(y_pred)].numpy()  # Ajustar o tamanho
    plotcharts(test_output[:len(y_pred)], y_pred, errors, nomes_test)  # Ajustar o tamanho
    torch.save(model.state_dict(), "modeloTreinado.pth")
    erro_pos_treinamento = erro_pos_treinamento.item() / len(test_output[:len(y_pred)])
    return tempo_total, erro_pos_treinamento, predicted, real

def plotcharts(test_output, y_pred, errors, nomes_test):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1)  # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(test_output.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    xx = [x for x in range(len(nomes_test))]
    for x, home, away in zip(xx, nomes_test['HomeTeam'], nomes_test['AwayTeam']):
        plt.text(x, 1.10, home, rotation='vertical', verticalalignment='bottom', horizontalalignment='center')    
    plt.xticks(xx, nomes_test['AwayTeam'], rotation='vertical')
    plt.legend(loc=0)
    plt.savefig('graficoTrain.png', bbox_inches='tight')

hidden_size = 500
# redeNeural("Teste", 0.9, 0.03, 20000, hidden_size, "D1")
import pandas as pd

def testar_modelo():
    df = pd.read_csv('D1.csv')
    df = df.replace(np.nan, 0)
    df2 = df.iloc[18:]
    df2['FTSG'] = df2['FTHG'] + df2['FTAG']
    df2['Over1.5'] = [1 if x > 1.5 else 0 for x in df2["FTSG"]]
    colunas = ['HomeTeam', 'AwayTeam', 'Over1.5', 'EGHT', 'EGAT',
               'BPHT', 'BPAT', 'GAHT', 'GAAT', 'SoGHT', 'SoGAT', 'FKHT', 'FKAT',
               'CKHT', 'CKAT', 'OHT', 'OAT', 'THT', 'TAT', 'GSHT', 'GSAT', 'FHT',
               'FAT', 'RCHT', 'RCAT', 'YCHT', 'YCAT', 'TPHT', 'TPAT', 'AHT', 'AAT',
               'DAHT', 'DAAT', 'CCHT', 'CCAT']
    df2 = df2[colunas]
    media_times = df.iloc[:9]
    media_times_com_over = media_times
    colunas.remove('Over1.5')
    media_times = media_times[colunas]
    media_times = media_times.groupby(['HomeTeam', 'AwayTeam']).mean()
    media_times.reset_index(inplace=True)
    test_input = media_times.iloc[:, 2:]
    scaler = StandardScaler()
    test_input = scaler.fit_transform(test_input)
    test_input = torch.tensor(test_input, dtype=torch.float32)
    input_size = test_input.size()[1]
    # Carregar o modelo treinado
    model = Net(input_size, hidden_size)
    model.load_state_dict(torch.load("modeloTreinado.pth"))
    model.eval()

    # Testar o modelo
    y_pred = model(test_input)
    predicted = y_pred.detach().numpy()
    real_values = df2['Over1.5'].values
    predicted_values = (predicted > 0.5).astype(int)
    
    # Calcular a acurácia
    accuracy = (predicted_values == real_values).mean()
    print(f'Acurácia: {accuracy * 100:.2f}%')
    
    # Exibir os valores reais e previstos
    min_length = min(len(media_times_com_over['HomeTeam']), len(media_times_com_over['AwayTeam']), len(real_values), len(predicted_values.flatten()))
    comparison_df = pd.DataFrame({
        'HomeTeam': media_times_com_over['HomeTeam'][:min_length],
        'AwayTeam': media_times_com_over['AwayTeam'][:min_length],
        'Real': real_values[:min_length],
        'Predicted': predicted_values.flatten()[:min_length],
        'Acertou': ["Sim" if x == 1 else "Não" for x in (predicted_values.flatten()[:min_length] == real_values[:min_length]).astype(int)]
    })
    print(comparison_df)


# Chamar a função para testar o modelo
testar_modelo()
