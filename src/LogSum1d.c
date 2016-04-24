#include "TH.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

struct LogAddFunctor : public thrust::binary_function<double,double,double> {
    const double NEGATIVE_INFINITY;
    LogAddFunctor(double _NEGATIVE_INFINITY) : NEGATIVE_INFINITY(_NEGATIVE_INFINITY) {}
    __host__ __device__ double operator() (double a, double b) const {
        if (a == NEGATIVE_INFINITY) return b;
        if (b == NEGATIVE_INFINITY) return a;
        return a>b ? a+log1p(exp(b-a)) : b+log1p(exp(a-b));
    }
};


static int jhu_THLogSum1d(lua_State *L) {
    THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
    long ndim = THDoubleTensor_nDimension(input);

    if(ndim != 1) {
        THError("input must be 1d");
    }

    const double NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();

    long N = THDoubleTensor_size(input, 0);
    double* array = THDoubleTensor_data(input);
    double result = thrust::reduce(array, array+N,
                                  NEGATIVE_INFINITY,
                                  LogAddFunctor(NEGATIVE_INFINITY));

    lua_pushnumber(L, result);

    return 1;
}

static const struct luaL_Reg jhu_THLogSum1d__ [] = {
    {"logsum1d", jhu_THLogSum1d},
    {0, 0}
};

static void jhu_THLogSum1d_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.DoubleTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THLogSum1d__, "jhu");
    lua_pop(L, 1);
}
