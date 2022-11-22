# audioprocessing
# some_file.py
import torch
from model import denoise_unet
from dataloader import SpeachData
from torch.utils.data import DataLoader

device = ('cuda' if torch.cuda.is_available() else 'cpu')

train_data = SpeachData(random_state=2)
test_data = SpeachData(random_state=2, test=True)
train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                              batch_size=128)
test_dataloader = DataLoader(test_data,
                             batch_size=128)

model = denoise_unet.UnetW().to(device)

model.train()
model.load_state_dict(torch.load('weights\\unetw.pt'))
loss_fn = torch.nn.L1Loss().to(device)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(
    [
        {"params": model.a, "lr": 1e-5},
        {"params": model.b, "lr": 1e-5},
        {"params": model.c, "lr": 1e-5},
        {"params": model.d, "lr": 1e-5},
        {"params": model.f, "lr": 1e-5},
        {"params": model.u1.parameters(), "lr": 5e-6},
        {"params": model.u2.parameters(), "lr": 5e-6},

    ],
    lr=1e-5,
)
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

    if (batch + 1) % 100 == 0:
        print(f"Looked at {batch * 128} samples")

    if (batch + 1) % 800 == 0:
        torch.save(model.state_dict(), 'denoiser_' + str(batch) + '.pt')
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for n, (X_t, y_t) in enumerate(test_dataloader):

                test_pred = model(X_t.to(device))

                test_loss += loss_fn(test_pred[:, :, 512:-512].to(device), y_t[:, :, 512:-512].to(device))

                if n > 25:
                    break

        print(f"Train Loss : {train_loss / 800}, \n Test loss: {test_loss / 25}")
        train_loss = 0
