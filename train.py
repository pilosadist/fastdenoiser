# audioprocessing
# some_file.py
import torch
from model import denoise_unet
from dataloader import SpeachData
from torch.utils.data import DataLoader

device = ('cuda' if torch.cuda.is_available() else 'cpu')

train_data = SpeachData(random_state=0)
test_data = SpeachData(random_state=0, test=True)
train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                              batch_size=512)
test_dataloader = DataLoader(test_data,
                             batch_size=512)

model = denoise_unet.Unet().to(device)

model.train()
model.load_state_dict(torch.load('best_w.pt'))
loss_fn = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
torch.manual_seed(0)

train_loss = 0
for batch, (X, y) in enumerate(train_dataloader):
    model.train()
    y_pred = model(X.to(device))

    loss = loss_fn(y_pred[:, :, 512:-512], y[:, :, 512:-512].to(device))

    train_loss += loss
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (batch + 1) % 50 == 0:
        print(f"Looked at {batch * 512} samples")
        torch.save(model.state_dict(), 'denoiser_' + str(batch) + '.pt')


    if (batch + 1) % 200 == 0:
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for n, (X_t, y_t) in enumerate(test_dataloader):

                test_pred = model(X_t.to(device))

                test_loss += loss_fn(test_pred[:, :, 512:-512].to(device), y_t[:, :, 512:-512].to(device))

                if n > 25:
                    break

        print(f"Train Loss : {train_loss / 200}, \n Test loss: {test_loss / 25}")
        train_loss = 0
