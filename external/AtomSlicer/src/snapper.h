// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include "mathutils.h"
#include "sdf.h"

namespace Snapper {
    void snap(std::vector<Vec3> &atoms, float layerHeight, const SDF &sdf, const std::vector<std::array<Vec3, 3>> &triangles);
}