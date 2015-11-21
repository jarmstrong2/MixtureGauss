require 'logdeterminant'
require 'optim'
require 'torch'
require 'nn'

det_size = 20^2 

l = nn.Linear(5, det_size)
reshape = nn.Reshape(20,20)
det = nn.LogDeterminant()

s = nn.Sequential()
s:add(l)
s:add(reshape)
s:add(det)

params, grad_params = s:getParameters()

input = torch.Tensor(2, 5)
input:uniform(-0.08, 0.08)

function feval(x)
	if x ~= params then
        	params:copy(x)
    	end
    	grad_params:zero()

	output = s:forward(input)
    doutput = s:backward(input, torch.ones(1, 20, 20))

	return output:sum(), grad_params	
end

diff, dC, dC_est = optim.checkgrad(feval, params, 1e-2)

print(output)
print(diff)
