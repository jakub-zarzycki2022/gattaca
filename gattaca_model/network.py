import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAT, EdgeConv
from torch_geometric.data.batch import Batch
import torch.optim as optim 
from torch.distributions import Categorical 


class GraphBranchingQNetwork(nn.Module):

    def __init__(self, state, action_space_dimension, number_of_actions):

        super().__init__()

        self.ac_dim = action_space_dimension
        self.n = number_of_actions

        self.state = state
        self.in_size = state * state


        self.conv_model1 = nn.Sequential(nn.Linear(2 * 2, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, state),
                                         )

        self.conv_model2 = nn.Sequential(nn.Linear(2 * state, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, state),
                                         )

        self.conv_model3 = nn.Sequential(nn.Linear(2 * state, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, state),
                                         )

        self.gconv1 = EdgeConv(self.conv_model1, aggr="add")
        self.gconv2 = EdgeConv(self.conv_model2, aggr="add")
        self.gconv3 = EdgeConv(self.conv_model3, aggr="add")
        # self.conv1 = nn.Conv1d(state, state, 3, padding=1, stride=1)
        # self.conv2 = nn.Conv1d(state, state, 3, padding=1, stride=1)

        self.model = nn.Sequential(nn.Linear(self.in_size, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   )

        self.pooling = nn.AvgPool1d(4)
        self.bn1 = nn.BatchNorm1d(state)
        self.bn2 = nn.BatchNorm1d(state)
        self.bn3 = nn.BatchNorm1d(state)

        self.value_head = nn.Sequential(nn.Linear(256, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 1)
                                        )

        self.adv_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_dimension)
        ) for _ in range(number_of_actions)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        out = self.gconv1(x, edge_index)
        out = F.relu(self.bn1(out))
        out = self.gconv2(out, edge_index)
        out = F.relu(self.bn2(out))
        out = self.gconv3(out, edge_index)
        out = F.relu(self.bn3(out))
        # out = self.pooling(out)

        out = out.reshape((x.shape[0], self.in_size))
        out = self.model(out)

        value = self.value_head(out)
        advantages = torch.stack([l(out) for l in self.adv_heads], dim=1)

        q_val = value.unsqueeze(2) + advantages - advantages.mean(2, keepdim=True)

        return q_val
