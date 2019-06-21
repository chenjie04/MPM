import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tcn import TemporalConvNet


class Multi_Preference_Model(nn.Module):
    def __init__(self, nb_users, nb_items, embed_dim, history_size):
        super(Multi_Preference_Model, self).__init__()

        self.nb_users = nb_users
        self.nb_items = nb_items
        self.embed_dim = embed_dim
        self.history_size = history_size

        #user and item embedding
        self.user_embed = nn.Embedding(self.nb_users, self.embed_dim)
        self.item_embed = nn.Embedding(self.nb_items, self.embed_dim)
        self.user_embed.weight.data.normal_(0., 0.01)
        self.item_embed.weight.data.normal_(0., 0.01)

        #TCN
        nhid = self.embed_dim
        level = 5
        num_channels = [nhid] * (level - 1) + [embed_dim]
        self.tcn = TemporalConvNet(num_inputs=self.embed_dim, num_channels=num_channels, kernel_size=3, dropout=0.25)

        #MLP
        mlp_layer_sizes = [self.embed_dim * 2, 128, 64, 32]
        nb_mlp_layers = len(mlp_layer_sizes)
        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i-1], mlp_layer_sizes[i])])

        #Output Module
        self.output_1 = nn.Linear(mlp_layer_sizes[-1] * (self.history_size + 1),128,bias=True)
        self.output_2 = nn.Linear(128,64,bias=True)
        self.output_3 = nn.Linear(64,32,bias=True)
        self.output_4 = nn.Linear(32,1,bias=True)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)

        lecunn_uniform(self.output_1)
        lecunn_uniform(self.output_2)
        lecunn_uniform(self.output_3)
        lecunn_uniform(self.output_4)

    def forward(self, user, item, history,sigmoid=False):

        item = self.item_embed(item)

        #multi granularity preference module
        xhistory = self.item_embed(history)

        output_TCN = self.tcn(xhistory.transpose(1,2)).transpose(1,2)

        predict_vectors = list()

        for i in range(self.history_size):
            preference = output_TCN[:, i, :]
            output_mlp = torch.cat((preference,item),dim=1)
            for j, layer in enumerate(self.mlp):
                output_mlp = layer(output_mlp)
                output_mlp = F.relu(output_mlp)

            output_mlp = output_mlp.view(-1, 1, output_mlp.size()[-1])
            predict_vectors.append(output_mlp)

        predict_vectors_sum = torch.cat(predict_vectors, dim=1)

        # general preference module
        user = self.user_embed(user)
        xmlp = torch.cat((user, item), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = F.relu(xmlp)

        #output module
        xmlp = xmlp.view(-1,1,xmlp.size()[-1])
        x = torch.cat((predict_vectors_sum,xmlp),dim=1)
        x = x.view(x.size()[0],-1)
        x = self.output_1(x)
        x = F.relu(x)
        x = self.output_2(x)
        x = F.relu(x)
        x = self.output_3(x)
        x = F.relu(x)
        x = self.output_4(x)

        if sigmoid:
            x = torch.sigmoid(x)
        return x