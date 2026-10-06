// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include "mathutils.h"
#include "sdf.h"
#include <vector>

class Toolpath {
public:
    struct WayPoint {
        Vec3 position;
        Vec3 normal;
        bool deposition;
    };

    struct Mesh {
        std::vector<float> vertices;
        std::vector<uint32_t> triangles;
    };

    Toolpath(const std::vector<Mesh> &layers, const SDF &sdf, const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float width, float height, uint32_t hardCodedField = 0, uint32_t zigzagOffset = 0, bool holeClosing = false, bool quiet = false);
    Toolpath() {}

    void smoothNormals(size_t iterations);
    void smoothPositions(size_t iterations, const SDF &sdf, float width, bool holeClosing);
    void singularityCone(float angle);
    void tessellateNormals(float angle);
    void saveToCSV(const char *path) const;
    void saveToPLY(const char *path, float layerHeight) const;

    std::vector<WayPoint> &getToolpath() { return toolpath; }
    const std::vector<WayPoint> &getToolpath() const { return toolpath; }

private:
    std::vector<WayPoint> toolpath;
    std::vector<std::array<float, 2>> statsToolpath;
    std::vector<float> statsLayer;
    std::vector<float> statsHeight;

    void directionsFromTexture(const char *path, std::vector<float> &directions, const std::vector<float> &positions, const Vec3 &modelSize) const;
};
