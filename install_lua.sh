#! /usr/bin/env bash

TH=`which th`
INSTALL=${TH%/bin/th}
JHU=$INSTALL/share/lua/5.1/jhu

cp Linear.lua $JHU
