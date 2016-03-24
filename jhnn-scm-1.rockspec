package = "jhnn"
version = "scm-1"

source = {
   url = "https://github.com/noa/jhnn.git",
}

description = {
   summary = "Torch extensions",
   detailed = [[]],
   homepage = "https://github.com/noa/jhnn",
   license = "MIT"
}

dependencies = {
   "torch",
   "nn"
}

build = {
  type = "command",
  build_command = [[
    cmake -E make_directory build;
    cd build;
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)";
    $(MAKE) -j12
  ]],
  install_command = "cd build && $(MAKE) install"
}
