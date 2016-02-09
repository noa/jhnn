package = "jhu"
version = "scm-1"

source = {
   url = "git://bitbucket.com/noandrews/icml2016.git",
}

description = {
   summary = "Torch extensions by researchers @ Johns Hopkins University",
   detailed = [[
   ]],
   homepage = "https://bitbucket.org/noandrews/jhu",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "totem"
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
