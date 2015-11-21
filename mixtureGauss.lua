require 'nn'
require 'nngraph'
require 'torch'
require 'logdeterminant'
require 'inverse_per_elem'

local mixture = {}

function mixture.gauss(inputSize, uDimSize, nMixture)
	target = nn.Identity()()
    pi = nn.Identity()()
    mu = nn.Identity()()
    u = nn.Identity()()

    u_reshaped = nn.Reshape(nMixture, uDimSize, inputSize)(u)
    u_pack = nn.SplitTable(2,4)(u_reshaped)
    mu_reshaped = nn.Reshape(nMixture, 1, inputSize)(mu)
    mu_pack = nn.SplitTable(2,4)(mu_reshaped)
    pi_reshaped = nn.Reshape(nMixture, 1)(pi)
    pi_pack = nn.SplitTable(2,3)(pi_reshaped)
    target_reshaped = nn.Reshape(1, inputSize)(target)

    for i = 1, nMixture do
        u_set = nn.SelectTable(i)(u_pack)
        mu_set = nn.SelectTable(i)(mu_pack)
        pi_set = nn.SelectTable(i)(pi_pack)

        sigma = nn.MM()({nn.Transpose({2,3})(u_set), u_set})

        det_sigma_2_pi = nn.Add(inputSize, inputSize * torch.log(2 * math.pi))
        (nn.LogDeterminant()(sigma))

        sqr_det_sigma_2_pi = nn.MulConstant(-0.5)(det_sigma_2_pi)

        target_mu = nn.CAddTable()({target_reshaped, nn.MulConstant(-1)(mu_set)})
        transpose_target_mu = nn.Transpose({2,3})(target_mu)
        inv_sigma = nn.Inverse()(sigma)
        transpose_target_mu_sigma = 
        nn.MM()({target_mu, inv_sigma})
        transpose_target_mu_sigma_target_mu = 
        nn.MM()({transpose_target_mu_sigma, transpose_target_mu})
        exp_term = nn.MulConstant(-0.5)(transpose_target_mu_sigma_target_mu)

        -- since we're using logsoftmax
        --log_pi = nn.Log(1)(pi_set)
        log_pi = pi_set

        mixture_result = nn.CAddTable()({log_pi, sqr_det_sigma_2_pi, exp_term})

        if i == 1 then
            add_mixture_result = nn.Exp()(mixture_result)
        else
            add_mixture_result = nn.CAddTable()({add_mixture_result, 
                nn.Exp()(mixture_result)})
        end

    end

    return nn.gModule({pi, mu, u, target}, {add_mixture_result})
end

return mixture
