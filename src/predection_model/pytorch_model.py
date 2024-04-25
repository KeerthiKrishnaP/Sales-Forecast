import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.data_models import ModelData


class NonLinearRegression(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):

        # Use super __init__ to inherit from parent nn.Module class
        super(NonLinearRegression, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc2(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc3(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc4(x)

        return x


def trian_model(
    model_data: ModelData,
    model: nn.Module,
    epochs: int,
    learing_rate: float,
    hidden_neurons: int,
) -> tuple[nn.Module,]:

    model = NonLinearRegression(1, hidden_neurons, 1)
    loss_function = nn.L1Loss()
    optimizer_function = torch.optim.SGD(params=model.parameters(), lr=learing_rate)
    X_train = torch.Tensor(np.asarray(model_data.train_data.x_train).reshape(-1, 1))
    y_train = torch.Tensor(np.asarray(model_data.train_data.y_train).reshape(-1, 1))
    X_test = torch.Tensor(np.asarray(model_data.test_data.x_test).reshape(-1, 1))
    y_test = torch.Tensor(np.asarray(model_data.test_data.y_test).reshape(-1, 1))
    epoch_count = []
    loss_values = []
    test_loss_values = []
    for epoch in range(epochs):
        # Train
        model.train()
        y_prediction = model.forward(X_train)
        loss = loss_function(y_prediction, y_train)
        optimizer_function.zero_grad()
        loss.backward()
        optimizer_function.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            test_preds = model(X_test)
            # calcaute loss
            test_loss = loss_function(test_preds, y_test)
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            # print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

    return model.eval(), test_preds.numpy(), loss_values
