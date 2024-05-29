import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y, z, label):
        alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
        label = label.view(-1, 1)

        # print("The o/p of first stream is:", x)
        # print("The o/p of second stream is:", y)
        # print("The o/p of third stream is:", z)

        # Ensure no NaN or Inf in the label
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     raise ValueError("x contains NaN or Inf values.")
        # elif torch.isnan(y).any() or torch.isinf(y).any():
        #     raise ValueError("y contains NaN or Inf values.")
        # elif torch.isnan(z).any() or torch.isinf(z).any():
        #     raise ValueError("z contains NaN or Inf values.")
        # else:
        #     pass

        # Ensure no NaN or Inf in the label
        if torch.isnan(label).any() or torch.isinf(label).any():
            raise ValueError("Label contains NaN or Inf values.")

        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)

        # Check for NaN or Inf in individual losses
        # if torch.isnan(loss_1).any() or torch.isinf(loss_1).any():
        #     raise ValueError("Loss_1 contains NaN or Inf values.")
        # if torch.isnan(loss_2).any() or torch.isinf(loss_2).any():
        #     raise ValueError("Loss_2 contains NaN or Inf values.")
        # if torch.isnan(loss_3).any() or torch.isinf(loss_3).any():
        #     raise ValueError("Loss_3 contains NaN or Inf values.")

        total_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + alpha_3 * loss_3
        return torch.mean(total_loss)


# Example usage:
if __name__ == '__main__':
    loss_fn = Loss()
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    z = torch.randn(10, 1)
    label = torch.randint(0, 2, (10, 1)).float()
    # print(label)
    loss = loss_fn(x, y, z, label)
    print(loss)
