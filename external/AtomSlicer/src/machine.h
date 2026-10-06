// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include "mathutils.h"
#include "toolpath.h"
#include <array>

namespace Machine {
    void addPlatformAndCenter(Toolpath &toolpath, float width, float height);
    void center3Axis(Toolpath &toolpath, float height);
    void toolpathToGCode(const Toolpath &toolpath, const char *path, float width, float height);
    void toolpathToGCode3Axis(const Toolpath &toolpath, const char *path, float width, float height);
};