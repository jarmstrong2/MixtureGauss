require 'logdeterminant'
require 'optim'
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
det_size = 20 


input = nn.Identity()()

l = nn.Linear(5, det_size)(input)
l_reshape = nn.Reshape(1,20)(l)
m = nn.MM()({nn.Transpose({2,3})(l_reshape), l_reshape})
det = nn.LogDeterminant()(m)
l2 = nn.Linear(1, 50)(det)

nng = nn.gModule({input}, {l2}):cuda()

params, grad_params = nng:getParameters()

input = torch.Tensor(2, 5)
input:uniform(-0.08, 0.08)

function feval(x)
	if x ~= params then
        	params:copy(x)
    	end
    	grad_params:zero()

	output = nng:forward(input:cuda())
    doutput = nng:backward(input:cuda(), torch.ones(2, 50):cuda())

	return output:sum(), grad_params:double()	
end

diff, dC, dC_est = optim.checkgrad(feval, params:double(), 1e-2)

print(output)
print(diff)
