#include "luaT.h"
#include "TH.h"

#include <lua.h>
#include <lualib.h>

LUA_EXTERNC DLL_EXPORT int luaopen_libjhu(lua_State *L);

int luaopen_libjhu(lua_State *L) {
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "jhu");
  
  return 1;
}
