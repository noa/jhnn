require 'jhnn'

local mytest = torch.TestSuite()
local tester = torch.Tester()

-- List of tests:
function mytest.LogSum()
   for _, ttype in pairs({'torch.FloatTensor'}) do
      -- 1D case
      local input  = torch.Tensor(10):type(ttype):normal(0, 1)
      local inputCopy = input:clone()
      local output = torch.Tensor(1):type(ttype)

      input:logsum(output)           -- call version 1
      local output2 = input:logsum() -- call version 2

      tester:eq(output, output2)
      tester:eq(input, inputCopy,1e-5)
      input:exp()
      local sum1 = input:sum()
      output:exp()
      local sum2 = output[1]
      local diff = sum1-sum2
      tester:eq(sum1, sum2, 0.1)
      
      -- 2D case (d1=batch, d2=dim)
      local D = 4
      local input  = torch.Tensor(D,D):type(ttype):normal(0, 1)
      local inputCopy = input:clone()
      local output = torch.Tensor(D)
      input:logsum(output)            -- call version 1
      local output2 = input:logsum()  -- call version 2
      tester:eq(output, output2)
      tester:eq(input, inputCopy, 1e-5) -- doesn't change input
      input:exp()
      output:exp()
      for b = 1, D do
         local sum1 = input[b]:sum()
         local sum2 = output[b]
         tester:eq(sum1, sum2, 0.1)
      end
   end
end

tester:add(mytest):run()
