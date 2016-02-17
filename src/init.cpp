#include "luaT.h"
#include "TH.h"
#include "THLogAdd.h"

#include "LogSum.c"
#include "LogSample.c"
#include "Sample.c"
#include "LogScale.c"
#include "Scale.c"

//#include "AlignmentFilter.cpp"

LUA_EXTERNC DLL_EXPORT int luaopen_libjhu(lua_State *L);

int luaopen_libjhu(lua_State *L) {
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_setglobal(L, "jhu");

    jhu_THLogSum_init(L);
    jhu_THLogSample_init(L);
    jhu_THSample_init(L);
    jhu_THLogScale_init(L);
    jhu_THScale_init(L);

    //luaopen_AlignmentFilter(L);

    return 1;
}