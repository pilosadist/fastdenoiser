# audioprocessing
# some_file.py
import torch
from model import net
from dataloader import SpeachData
from torch.utils.data import DataLoader

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# load data
train_data = SpeachData(random_state=0)
test_data = SpeachData(random_state=0, test=True)
train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                              batch_size=512)
test_dataloader = DataLoader(test_data,
                             batch_size=512)

# model_initialization
model = net.Denoiser().to(device)

# model.train()
# model.load_state_dict(torch.load('unet_model_in_epoch_3.pt'))
loss_fn = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
torch.manual_seed(0)

# Set the number of epochs (we'll keep this small for faster training times)
train_loss = 0
for batch, (X, y) in enumerate(train_dataloader):
    model.train()
    # 1. Forward pass
    X = torch.cat([X, X], dim=1)
    y_pred = model(X.to(device))[:, 0:1, :]
    # 2. Calculate loss (per batch)
    loss = loss_fn(y_pred[:, :, 512:-512], y[:, :, 512:-512].to(device))
    train_loss += loss  # accumulatively add up the loss per epoch
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Print out how many samples have been seen
    if (batch + 1) % 50 == 0:
        print(f"Looked at {batch * 512} samples")
        torch.save(model.state_dict(), 'denoiser_' + str(batch) + '.pt')

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)

    if (batch + 1) % 200 == 0:
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for n, (X_t, y_t) in enumerate(test_dataloader):
                # 1. Forward pass
                X_t = torch.cat([X_t, X_t], dim=1)

                test_pred = model(X_t.to(device))[:, 0:1, :]

                # 2. Calculate loss (accumatively)
                test_loss += loss_fn(test_pred[:, :, 512:-512].to(device), y_t[:, :, 512:-512].to(device))

                if n > 25:
                    break

        print(f"Train Loss : {train_loss / 200}, \n Test loss: {test_loss / 25}")
        train_loss = 0
