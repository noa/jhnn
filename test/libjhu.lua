-- Tester:
require 'torch'
require 'libjhu'
local totem = require 'totem'
local tester = totem.Tester()

local function maxdiff(x, y)
   local d = x-y
   if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.FloatTensor' then
      return d:abs():max()
   else
      local dd = torch.Tensor():resize(d:size()):copy(d)
      return dd:abs():max()
   end
end

-- List of tests:
local tests = {
   LogSum = function()
      -- 1D case
      local input  = torch.DoubleTensor(10):normal(0, 1)
      local output = torch.DoubleTensor(1)
      input.jhu.logsum(input, output)
      input:exp()
      local sum1 = input:sum()
      output:exp()
      local sum2 = output[1]
      local diff = sum1-sum2
      tester:assert(math.abs(diff) < 1e-2, 'bad log sum: err='..diff)

      -- 2D case (d1=batch, d2=dim)
      local D = 4
      local input  = torch.DoubleTensor(D,D):normal(0, 1)
      local output = torch.DoubleTensor(D)
      input.jhu.logsum(input, output)
      input:exp()
      output:exp()
      for b = 1, D do
         local sum1 = input[b]:sum()
         local sum2 = output[b]
         local diff = sum1-sum2
         tester:assert(math.abs(diff) < 1e-2, 'bad log sum: err='..diff)
      end
   end,
   LogSample1D = function()
      local D = 10
      local N = 50000
      local P = torch.DoubleTensor(D):uniform(0, 1)
      local N1 = torch.multinomial(P, N, true)
      local S1 = N1:double():sum() / N
      local logP = P:log()
      local tmp = torch.zeros(1):double()
      local N2 = torch.zeros(1):double()
      for n = 1, N do
         tmp.jhu.logsample(logP:clone(), tmp)
         N2:add(tmp)
      end
      local S2 = N2[1] / N
      local diff = math.abs(S1-S2)
      tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
   end,
   Sample1D = function()
      local D = 10
      local N = 50000
      local P = torch.DoubleTensor(D):uniform(0, 1)
      local N1 = torch.multinomial(P, N, true)
      local S1 = N1:double():sum() / N
      local tmp = torch.zeros(1):double()
      local N2 = torch.zeros(1):double()
      for n = 1, N do
         tmp.jhu.sample(P:clone(), tmp)
         N2:add(tmp)
      end
      local S2 = N2[1] / N
      local diff = math.abs(S1-S2)
      tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
   end,
   LogSampleEdgeCase = function()
      local lnP = torch.DoubleTensor({-math.huge, -math.huge, -math.huge, -math.huge, 0})
      local result = torch.DoubleTensor(1)
      result.jhu.logsample(lnP, result)
      tester:assert(result[1] == 5, "error with edge case")
   end,
   LogSampleNormalized = function()
      local D = 10
      local N = 50000
      local P = torch.DoubleTensor(D, D):uniform(0, 1)
      for d = 1, D do
         local Z = P[d]:sum()
         P[d]:div(Z)
      end
      
      -- Take some samples for the distributions in probability space:
      local N1 = torch.multinomial(P, N, true)
      local S1 = N1:double():sum(2):div(N)

      -- Now convert to log space
      local logP = torch.log(P)
      local tmp = torch.zeros(D):double()
      local N2 = torch.zeros(D):double()
      for n = 1, N do
         tmp.jhu.logsample(logP:clone(), tmp)
         N2:add(tmp)
      end
      local S2 = N2:div(N)

      local diff = maxdiff(S1,S2)
      tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
   end,
   SampleNormalized = function()
      local D = 10
      local N = 50000
      local P = torch.DoubleTensor(D, D):uniform(0, 1)
      for d = 1, D do
         local Z = P[d]:sum()
         P[d]:div(Z)
      end
      
      -- Take some samples for the distributions in probability space:
      local N1 = torch.multinomial(P, N, true)
      local S1 = N1:double():sum(2):div(N)

      -- Now convert to log space
      local tmp = torch.zeros(D):double()
      local N2 = torch.zeros(D):double()
      for n = 1, N do
         tmp.jhu.sample(P:clone(), tmp)
         N2:add(tmp)
      end
      local S2 = N2:div(N)
      local diff = maxdiff(S1,S2)
      tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
   end,
   LogSampleUnnormalized = function()
      local D = 10
      local N = 50000
      local P = torch.DoubleTensor(D, D):uniform(0, 1)

      -- Take some samples for the distributions in probability space:
      local N1 = torch.multinomial(P, N, true)
      local S1 = N1:double():sum(2):div(N)

      -- Now convert to log space
      local logP = torch.log(P)
      local tmp = torch.zeros(D):double()
      local N2 = torch.zeros(D):double()
      for n = 1, N do
         tmp.jhu.logsample(logP:clone(), tmp)
         N2:add(tmp)
      end
      local S2 = N2:div(N)

      local diff = maxdiff(S1,S2)
      tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
   end,
   SampleUnnormalized = function()
      local D = 10
      local N = 50000
      local P = torch.DoubleTensor(D, D):uniform(0, 1)

      -- Take some samples for the distributions in probability space:
      local N1 = torch.multinomial(P, N, true)
      local S1 = N1:double():sum(2):div(N)

      -- Now convert to log space
      local tmp = torch.zeros(D):double()
      local N2 = torch.zeros(D):double()
      for n = 1, N do
         tmp.jhu.sample(P:clone(), tmp)
         N2:add(tmp)
      end
      local S2 = N2:div(N)

      local diff = maxdiff(S1,S2)
      tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
   end,
   LogScale = function()
      local D = 10
      -- Vector
      local P = torch.DoubleTensor(D):uniform(0, 1)
      local logP = torch.log(P)
      logP.jhu.logscale(logP)
      local diff = math.abs(logP:sum()-1.0)
      tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
      
      -- Matrix
      local P = torch.DoubleTensor(D, D):uniform(0, 1)
      local logP = torch.log(P)
      logP.jhu.logscale(logP)
      local diff = math.abs(logP:sum()-D)
      tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
   end,
   Scale = function()
      local D = 10
      -- Vector
      local P = torch.DoubleTensor(D):uniform(0, 1)
      P.jhu.scale(P)
      local diff = math.abs(P:sum()-1.0)
      tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
      
      -- Matrix
      local P = torch.DoubleTensor(D, D):uniform(0, 1)
      P.jhu.logscale(P)
      local diff = math.abs(P:sum()-D)
      tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
   end
}

-- Run the tests:
tester:add(tests):run()
