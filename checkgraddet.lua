require 'logdeterminant'
require 'optim'
require 'torch'
require 'nn'

det_size = 20^2 

l = nn.Linear(5, det_size)
det = nn.LogDeterminant()

s = nn.Sequential():cuda()
s:add(l)
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

	return doutput:sum(), grad_params	
end

diff, dC, dC_est = optim.checkgrad(feval, params, 1e-2)

print(output)
print(diff)
