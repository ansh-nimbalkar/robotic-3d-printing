// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once

#include <set>
#include <array>
#include <cstdint>

namespace Border {
std::set<std::array<uint32_t, 2>> borderEdges(const uint32_t *triangles, uint32_t triangleCounts);
}