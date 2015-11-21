require 'nn'

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
        input_single_resize = input_inv:clone():resize(1, 1, inputSize^2)
        input_single_expand = input_single_resize:expand(inputSize, inputSize, inputSize^2)
        input_entire_resize = input_inv:clone():resize(inputSize, inputSize, 1)
        input_entire_expand = input_entire_resize:expand(inputSize, inputSize, inputSize^2)

        input_cmul = torch.cmul(input_single_expand, input_entire_expand)
        input_sum = input_cmul:sum(1):sum(2)
        input_sum:resize(inputSize, inputSize)
        input_sum:mul(-1)
        self.gradInput[i] = input_sum       

        self.gradInput[i]:cmul(gradOutput[i])
    end
    return self.gradInput
end