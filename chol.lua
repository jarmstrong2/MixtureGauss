function cholesky (A)
  local n = A:size(1)
  --assert(A:size"#" == 2 and A:size(2) == n, "square matrix expected")
  local U = torch.zeros(n, n)
  local C = {} -- columns of U
  for i = 1, n do C[i] = U[{{},{i}}] end
  -- factorization:
  -- i = 1
  local Ai, Ui = A[1], U[1]
  local uii = torch.sqrt(Ai[1])
  Ui[1] = uii
  for j = 2, n do Ui[j] = Ai[j] / uii end
  -- i > 1
  for i = 2, n do
    Ai, Ui = A[i], U[i]
    local uci = C[i](1, i - 1) -- U(i, 1:(i-1))
    uii = torch.sqrt(Ai[i] - torch.dot(uci, uci))
    Ui[i] = uii
    for j = i + 1, n do
      local ucj = C[j](1, i - 1) -- U(j, 1:(i-1))
      Ui[j] = (Ai[j] - torch.dot(uci, ucj)) / uii
    end
  end
  return U
end
