require 'torch'

function adjugate(mat)
	matSize = mat:size()[1]
	newmat = torch.zeros(matSize,matSize):double()
	for i=0,matSize-1 do
		if (i % 2 ~= 0) then
			multiple = -1
		else
			multiple = 1
		end
		if i == 0 then
			diag = mat:diag()
			diagmat = torch.diag(diag)
			newmat = diagmat 
		else
			diag1 = mat:diag(i):mul(multiple)
			diag2 = mat:diag(-i):mul(multiple)
			diag1mat = torch.diag(diag1, i)
			diag2mat = torch.diag(diag2, -i)
			newmat = newmat + diag1mat + diag2mat
		end
    end
    return newmat
end