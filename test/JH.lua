-- luacheck: globals torch

-- Tester:
require 'torch'
require 'jhu'

local mytest = torch.TestSuite()
local tester = torch.Tester()

-- List of tests:
function mytest.LogScale()
   local types = { 'torch.DoubleTensor', 'torch.FloatTensor' }
   for _, t in pairs(types) do 
      local D = 10
      -- Vector
      local P = torch.Tensor(D):type(t)
      P:uniform(0, 1)
      local logP = torch.log(P)
      local logP_clone = P:clone()
      jhu.logscale(logP)
      local diff = math.abs(logP:sum()-1.0)
      tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
      logP_clone:logscale()
      tester:eq(logP_clone, logP, 1e-3)
      
      -- Matrix
      local P = torch.Tensor(D, D):type(t)
      P:uniform(0, 1)
      local logP = torch.log(P)
      local logP_clone = logP:clone()
      jhu.logscale(logP)
      local diff = math.abs(logP:sum()-D)
      tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
      logP_clone:logscale()
      tester:eq(logP_clone, logP, 1e-3)
   end
end

tester:add(mytest):run()
