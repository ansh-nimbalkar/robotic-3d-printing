// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once

#include "sdf.h"
#include <string>

namespace Infill {
enum Type {
    Full = 0,
    Hollow = 1,
    Grid = 2,
    Cubic = 3,
    Gyroid = 4,
};

std::string infillToString(Type infill);

void generate(SDF &sdf, Type type, float depositionWidth);
};