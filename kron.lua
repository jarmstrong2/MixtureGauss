function kron(A,B)
    local m, n = A:size(1), A:size(2)
    local p, q = B:size(1), B:size(2)
    local C = torch.Tensor(m*p,n*q)

    for i=1,m do
        for j=1,n do
            C[{{(i-1)*p+1,i*p},{(j-1)*q+1,j*q}}] = torch.mul(B, A[i][j])
        end
    end
    return C
end