require 'nn'
require 'kron'
local Inverse, parent = torch.class('nn.Inverse', 'nn.Module')


function Inverse:updateOutput(input)
    batchSize = input:size()[1]
    dim = input:size()[2]
    self.output = torch.zeros(batchSize, dim, dim)
    for i = 1, batchSize do
        inputSize = ((input[i]):size())[1]
        eps = torch.eye(inputSize) * 1e-2
        self.output[i] = torch.inverse(input[i] + eps)
    end
    return self.output
end

function Inverse:updateGradInput(input, gradOutput)
    batchSize = input:size()[1]
    dim = input:size()[2]
    self.gradInput =  torch.zeros(batchSize, dim, dim)
    for i = 1, batchSize do
        inputSize = dim
        eps = torch.eye(inputSize) * 1e-2  
        input_inv = torch.inverse(input[i] + eps)
        input_t_inv = torch.inverse((input[i] + eps):t())
        kron_prod = -kron(input_t_inv, input_inv)
        kron_sum = torch.sum(kron_prod,2):resize(dim,dim)
        self.gradInput[i] = kron_sum 
    end
    return self.gradInput
end