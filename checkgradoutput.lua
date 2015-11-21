require 'yHat'
require 'mixtureGauss'
require 'optim'
require 'torch'
require 'nn'
require 'cunn'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-inputSize' , 30, 'number of input dimension')
cmd:option('-epsilon' , 1e-5, 'number of input dimension')
cmd:option('-hiddenSize' , 400, 'number of hidden units in lstms')
cmd:option('-lr' , 1e-5, 'learning rate')
cmd:option('-maxlen' , 2, 'max sequence length')
cmd:option('-batchSize' , 4, 'mini batch size')
cmd:option('-numPasses' , 4, 'number of passes')
cmd:option('-isCovarianceFull' , true, 'true if full covariance, o.w. diagonal covariance')
cmd:option('-numMixture' , 1, 'number of mixture components in output layer')
cmd:option('-dimSize' , 1, 'number of mixture components in output layer') 

cmd:text()
opt = cmd:parse(arg)

mix = require 'mixtureGauss'
gauss = mix.gauss(opt.inputSize, opt.dimSize, opt.numMixture)

y_size = opt.numMixture + (opt.inputSize * opt.numMixture) + (opt.inputSize * opt.numMixture * opt.dimSize) 

l = nn.Linear(5, y_size):cuda()
y = nn.YHat():cuda()

s = nn.Sequential():cuda()
s:add(l)
s:add(y)

params, grad_params = s:getParameters()

print(params:size())

--mask = torch.ones(2, 1):cuda()
--mixture:setmask(mask)

input = torch.CudaTensor(2, 5)
input:uniform(-0.08, 0.08)

target = torch.randn(2, opt.inputSize):cuda()

function feval(x)
	if x ~= params then
        	params:copy(x)
    	end
    	grad_params:zero()

	output = s:forward(input)
	a,b,c = unpack(output)
	loss = gauss:forward({a:float(),b:float(),c:float(),target:float()})
	print(a)
    --loss = output:sum()
	mixgrad = gauss:backward({a:float(),b:float(),c:float(),target:float()},torch.ones(2,1):float())
	grad_y = s:backward(input, mixgrad)
	--grad_y = s:backward(input, output:clone():fill(1):cuda())

	return loss, grad_params:double() 	
end

diff, dC, dC_est = optim.checkgrad(feval, params, opt.epsilon)

print(output)
print(diff)
