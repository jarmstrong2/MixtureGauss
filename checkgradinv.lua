require 'inverse2'
require 'optim'
require 'torch'
require 'nn'
require 'nngraph'

det_size = 20 


input = nn.Identity()()

l = nn.Linear(5, det_size)(input)
l_reshape = nn.Reshape(1,20)(l)
m = nn.MM()({nn.Transpose({2,3})(l_reshape), l_reshape})
inv = nn.Inverse()(m)
inv_reshape = nn.Reshape(20*20)(inv)
l2 = nn.Linear(20*20, 10)(inv_reshape)

nng = nn.gModule({input}, {l2})

params, grad_params = nng:getParameters()

input = torch.Tensor(2, 5)
input:uniform(-0.08, 0.08)

function feval(x)
	if x ~= params then
        	params:copy(x)
    	end
    	grad_params:zero()

	output = nng:forward(input)
    doutput = nng:backward(input, torch.ones(2,10))

	return output:sum(), grad_params	
end

diff, dC, dC_est = optim.checkgrad(feval, params, 1e-2)

print(output)
print(diff)
