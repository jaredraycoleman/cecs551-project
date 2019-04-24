#from gcn.layers import *
#from gcn.metrics import *
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, inputs, outputs, support, dropout,
            sparse_inputs, activation, bias, 
            placeholders, input_dim, learning_rate, weight_decay):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer, inputs, outputs, support, 
            dropout=True, sparse_inputs=True, F.relu, bias=True)
        self.layers.append(GCNLayer, inputs, outputs, support,
            dropout=True, sparse_inputs=True, None, bias=True)

        self.optimizer = torch.optim.Adam(lr = 0.1)
        def forward(self, features):
            h = features
            for layer in self.layers:
                h = layer(h)

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += weight_decay + MSELoss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
