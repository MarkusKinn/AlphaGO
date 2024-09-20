import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation_fn=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.activation_fn(x)
        return x

class ANET(pl.LightningModule):
    def __init__(self, config):
        super(ANET, self).__init__()
        self.save_hyperparameters()
        self.config = config

        self.board_size = config.getint("game", "board_size")
        activation_fn = getattr(nn, config.get("architecture", "activation_function"))()
        num_residual_blocks = config.getint("architecture", "num_residual_blocks")
        num_channels = config.getint("architecture", "num_channels")

        self.features = nn.Sequential(
            nn.Conv2d(4, num_channels, kernel_size=3, padding=1),
            activation_fn
        )

        for _ in range(num_residual_blocks):
            self.features.add_module(f"residual_block_{_}", ResidualBlock(num_channels, num_channels))
            self.features.add_module(f"batch_norm_{_}", nn.BatchNorm2d(num_channels))
            self.features.add_module(f"activation_{_}", activation_fn)

        # Fully Connected Layers
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            activation_fn,
            nn.Dropout(0.2),
            nn.Linear(256, self.board_size ** 2),
        )
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.policy_head(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config.get("training", "lr")))
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.to(self.device))
        loss = self.loss_fn(outputs, targets.to(self.device))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return loss

    def train_dataloader(self):
        input_tensors = []
        target_tensors = []

        for state_tensor, visit_counts in self.training_data:
            target_tensor = torch.tensor([visit_counts.get(action, 0.0) for action in visit_counts.keys()],
                                         dtype=torch.float32)
            input_tensors.append(state_tensor)
            target_tensors.append(target_tensor)

        input_tensor = torch.stack(input_tensors)
        target_tensor = torch.stack(target_tensors)

        dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.getint("training", "batch_size"), shuffle=True)

        return train_loader

    def save_weights(self, file_path):
        """
        Save the current weights of the ANET model.

        Args:
            file_path (str): The path to save the model weights.
        """
        torch.save(self.state_dict(), file_path)

    def load_weights(self, file_path):
        """
        Load the ANET model weights from a file.

        Args:
            file_path (str): The path to load the model weights from.
        """
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def predict(self, inputs, possible_actions, stochastic=True):
        self.eval()
        with torch.no_grad():
            logits = self(inputs)
            action_probs = F.softmax(logits, dim=1)

            # Set the probability of invalid actions to zero
            valid_action_indices = torch.tensor([x * self.board_size + y for x, y in possible_actions], dtype=torch.long)
            invalid_action_indices = ~torch.isin(torch.arange(self.board_size ** 2), valid_action_indices)
            action_probs[:, invalid_action_indices] = 0.0

            # Renormalize the probabilities
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)

            if stochastic:
                # Sample an action based on the probability distribution
                action_idx = torch.multinomial(action_probs, num_samples=1).item()
                action_x, action_y = divmod(action_idx, self.board_size)
                action = (action_x, action_y)
                return action, action_probs[0, action_idx].item()
            else:
                # Select the action with the maximum probability
                max_prob, max_idx = torch.max(action_probs, dim=1)
                action_x, action_y = divmod(max_idx.item(), self.board_size)
                best_action = (action_x, action_y)
                return best_action, max_prob.item()