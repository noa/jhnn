-- luacheck: globals torch

-- Tester:
require 'torch'
require 'libjhu'

local mytest = torch.TestSuite()
local tester = torch.Tester()

-- List of tests:
function mytest.LogSum()
   -- 1D case
   local input  = torch.DoubleTensor(10):normal(0, 1)
   local inputCopy = input:clone()
   local output = torch.DoubleTensor(1)
   input.jhu.logsum(input, output)
   tester:eq(input, inputCopy,1e-5)
   input:exp()
   local sum1 = input:sum()
   output:exp()
   local sum2 = output[1]
   local diff = sum1-sum2
   tester:eq(sum1, sum2, 0.1)

   -- 2D case (d1=batch, d2=dim)
   local D = 4
   local input  = torch.DoubleTensor(D,D):normal(0, 1)
   local inputCopy = input:clone()
   local output = torch.DoubleTensor(D)
   input.jhu.logsum(input, output)
   tester:eq(input, inputCopy, 1e-5)
   input:exp()
   output:exp()
   for b = 1, D do
      local sum1 = input[b]:sum()
      local sum2 = output[b]
      tester:eq(sum1, sum2, 0.1)
   end
end

function mytest.LogSample1D()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D):uniform(0, 1)
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum() / N
   local logP = P:log()
   -- local tmp = torch.zeros(1):double()
   -- local N2 = torch.zeros(1):double()
   local tmp = torch.LongTensor(1):zero()
   local N2 = torch.LongTensor(1):zero()
   for n = 1, N do
      logP.jhu.logsample(logP:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2[1] / N
   tester:eq(S1, S2, 0.1)
end

function mytest.Sample1D()
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
   tester:eq(S1, S2, 0.1)
end

function mytest.LogSampleEdgeCase()
   local lnP = torch.DoubleTensor({-math.huge, -math.huge, -math.huge, -math.huge, 0})
   --local result = torch.DoubleTensor(1)
   local result = torch.LongTensor(1)
   lnP.jhu.logsample(lnP, result)
   tester:eq(result[1],5)
end

function mytest.LogSampleEdgeCaseTwo()
   local x = torch.DoubleTensor({-math.huge, -math.huge, -math.huge, -0.5})
   local y = torch.DoubleTensor(1,1)
   local x = x:view(1,4)
   x.jhu.logsum(x,y)
   local z = y[1][1]
   assert(type(z) == 'number')
   assert(not (z ~= z), "NaN result!") -- check not NaN
end

function mytest.LogSampleNormalized()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D, D):uniform(0, 1)
   for d = 1, D do
      local Z = P[d]:sum()
      P[d]:div(Z)
   end

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)

   -- Now convert to log space
   local logP = torch.log(P)
   -- local tmp = torch.zeros(D):long()
   local N2 = torch.zeros(D)
   local tmp = torch.LongTensor(D):zero()
   --local N2 = torch.LongTensor(D):zero()
   for n = 1, N do
      logP.jhu.logsample(logP:clone(), tmp)
      N2:add(tmp:double())
   end
   local S2 = N2:div(N)

   -- print('S1')
   -- print(S1)
   -- print('S2')
   -- print(S2)

   tester:eq(S1, S2, 0.1)
end

function mytest.SampleNormalized()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D, D):uniform(0, 1)
   for d = 1, D do
      local Z = P[d]:sum()
      P[d]:div(Z)
   end

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)

   -- Now convert to log space
   local tmp = torch.zeros(D):double()
   local N2 = torch.zeros(D):double()
   for n = 1, N do
      tmp.jhu.sample(P:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2:div(N)
   tester:eq(S1, S2, 0.1)
end

function mytest.LogSampleUnnormalized()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D, D):uniform(0, 1)

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)

   -- Now convert to log space
   local logP = torch.log(P)
   -- local tmp = torch.zeros(D):double()
   local N2 = torch.zeros(D)
   local tmp = torch.LongTensor(D):zero()
   --local N2 = torch.LongTensor(D):zero()
   for n = 1, N do
      logP.jhu.logsample(logP:clone(), tmp)
      N2:add(tmp:double())
   end
   local S2 = N2:div(N)
   tester:eq(S1,S2,0.1)
end

function mytest.EncodeDecode()
   local N = 7
   local dim = { 64, 128, 256, 512, 1024, 2048 }

   local function encode(i, j, result)
      result:copy(i)
      return result:map(j, function(s, t) return s + (t-1)*N end)
   end

   local function decode(i, o1, o2)
      for k = 1, i:size(1) do
         o1[k] = ((i[k]-1) % N) + 1
         o2[k] = math.floor(((i[k]-1) / N) + 1)
      end
   end

   for _, d in ipairs(dim) do
      local input1 = torch.LongTensor(d):random(7)
      local input2 = torch.LongTensor(d):random(7)

      local t = torch.tic()
      local gold = torch.LongTensor(d)
      encode(input1, input2, gold)
      local elapsed1 = torch.toc(t)

      local result = torch.LongTensor(d)
      t = torch.tic()
      result.jhu.encode(input1, input2, result, N)
      local elapsed2 = torch.toc(t)

      tester:eq(gold, result)
      tester:assertlt(elapsed2, elapsed1, "too slow")

      -- decode to get inputs back
      local decoded1 = torch.LongTensor(d)
      local decoded2 = torch.LongTensor(d)

      local t = torch.tic()
      decode(result, decoded1, decoded2)
      local elapsed1 = torch.toc(t)

      tester:eq(decoded1, input1)
      tester:eq(decoded2, input2)

      local t = torch.tic()
      result.jhu.decode(result, decoded1, decoded2, N)
      local elapsed2 = torch.toc(t)
      tester:assertlt(elapsed2, elapsed1, "too slow")

      tester:eq(decoded1, input1)
      tester:eq(decoded2, input2)
   end
end

function mytest.SampleUnnormalized()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D, D):uniform(0, 1)

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)

   -- Now convert to log space
   local tmp = torch.zeros(D):double()
   local N2 = torch.zeros(D):double()
   for n = 1, N do
      tmp.jhu.sample(P:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2:div(N)

   tester:eq(S1, S2, 0.1)
end

function mytest.LogScale()
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
end

function mytest.Scale()
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

tester:add(mytest):run()
