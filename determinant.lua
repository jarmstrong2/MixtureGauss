require 'nn'

local Determinant, parent = torch.class('nn.Determinant', 'nn.Module')

function Determinant:updateOutput(input)
    batchSize = input:size()[1]
    self.output = torch.zeros(batchSize, 1)
    for i = 1, batchSize do
        eig_vals = torch.eig(input[i], 'N')
        self.output[i] = eig_vals:select(2, 1):prod()
    end
    return self.output
end

function Determinant:updateGradInput(input, gradOutput)
    batchSize = input:size()[1]
    self.gradInput =  torch.zeros(input:size())
    for i = 1, batchSize do  
        eig_vals = torch.eig(input[i], 'N')
        detInput = eig_vals:select(2, 1):prod()
        invInput = torch.inverse(input[i])
        adjInput = invInput * detInput
        self.gradInput[i] = adjInput:t() * gradOutput[i]
    end
    return self.gradInput
end