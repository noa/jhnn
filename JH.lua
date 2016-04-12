local ffi = require 'ffi'

local JH = {}

local generic_JH_h = require 'jhnn.JH_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in JH.h
generic_JH_h = generic_JH_h:gsub("\n#[^\n]*", "")
generic_JH_h = generic_JH_h:gsub("^#[^\n]*\n", "")

-- THGenerator struct declaration copied from torch7/lib/TH/THRandom.h
local base_declarations = [[
typedef void JHState;

typedef struct {
  unsigned long the_initial_seed;
  int left;
  int seeded;
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;
]]

-- polyfill for LUA 5.1
if not package.searchpath then
   local sep = package.config:sub(1,1)
   function package.searchpath(mod, path)
      mod = mod:gsub('%.', sep)
      for m in path:gmatch('[^;]+') do
         local nm = m:gsub('?', mod)
         local f = io.open(nm, 'r')
         if f then
            f:close()
            return nm
         end
     end
   end
end

-- load libJH
JH.C = ffi.load(package.searchpath('libJH', package.cpath))

ffi.cdef(base_declarations)

-- expand macros, allow to use original lines from lib/JH/generic/JH.h
local preprocessed = string.gsub(generic_JH_h, 'TH_API void JH_%(([%a%d_]+)%)', 'void JH_TYPE%1')

local replacements =
{
   {
      ['TYPE'] = 'Double',
      ['real'] = 'double',
      ['THTensor'] = 'THDoubleTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
   },
   {
      ['TYPE'] = 'Float',
      ['real'] = 'float',
      ['THTensor'] = 'THFloatTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
    }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

JH.NULL = ffi.NULL or nil

function JH.getState()
   return ffi.NULL or nil
end

function JH.optionalTensor(t)
   return t and t:cdata() or JH.NULL
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void JH_%(([%a%d_]+)%)') do
      t[#t+1] = n
   end
   return t
end

function JH.bind(lib, base_names, type_name)
   local ftable = {}
   local prefix = 'JH_' .. type_name
   for _,n in ipairs(base_names) do
      -- use pcall since some libs might not support all functions
      -- (e.g. cunn). The pcall function calls its first argument in
      -- protected mode, so that it catches any errors while the
      -- function is running. If there are no errors, pcall returns
      -- true, plus any values returned by the call. Otherwise, it
      -- returns false, plus the error message.
      local ok,v = pcall(function() return lib[prefix .. n] end)
      if ok then
         ftable[n] = function(...) v(...) end -- implicitely add state
      else
         print('not found: ' .. prefix .. n .. v)
      end
   end
   return ftable
end

-- build function table from the definitions in JH.h
local function_names = extract_function_names(generic_JH_h)

print('[JH] function names:')
print(function_names)

JH.kernels = {}
JH.kernels['torch.FloatTensor']  = JH.bind(JH.C, function_names, 'Float', JH.getState)
JH.kernels['torch.DoubleTensor'] = JH.bind(JH.C, function_names, 'Double', JH.getState)

for _, type in pairs({'torch.FloatTensor', 'torch.DoubleTensor'}) do
   local mt = torch.getmetatable(type)
   for k,v in pairs(JH.kernels[type]) do
      mt[k] = v
   end
end

return JH
