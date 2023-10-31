import torch

class DenseLayer:
  def __init__(self, n_features, n_neurons):
    self.weights = 0.01 * torch.rand((n_features, n_neurons))
    self.biases = torch.zeros((1, n_neurons))

  def forward(self, inputs):
    self.output = torch.matmul(inputs, self.weights) + self.biases
  
if __name__ == "__main__":
    layer1 = DenseLayer(5, 16)
    layer2 = DenseLayer(16, 16)
    layer3 = DenseLayer(16, 16)
    output_layer = DenseLayer(16, 5)

    X = torch.rand((32, 5))
    layer1.forward(X)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    output_layer.forward(layer3.output)
    print("Output shape: ", output_layer.output.shape)