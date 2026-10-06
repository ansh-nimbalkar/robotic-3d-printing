// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once

#include "toolpath.h"
#include "infill.h"
#include "normal.h"
#include <array>

class Slicer {
public:
    Slicer(const std::vector<std::array<Vec3, 3>> &triangles, float nozzleWidth, float layerHeight, float maxTopAngle, float maxBottomAngle, Infill::Type infillType, NormalField::Objective objective, uint32_t hardCodedField = 0, float numberOfCover = 4.0f, uint32_t zigzagOffset = 0, float nozzleConeAngle = 95.0f, bool upCollisionCheck = false, bool holeClosing = false);    
    void saveToGCode(const char *path) const;
    void saveToGCode3Axis(const char *path) const;
    void saveToPLY(const char *path) const;

private:
    Toolpath toolpath;
    float layerHeight;
};