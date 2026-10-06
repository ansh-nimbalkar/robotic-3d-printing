// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "snapper.h"
#include "bvhtriangle.h"
#include "parallel.h"

void Snapper::snap(std::vector<Vec3> &atoms, float layerHeight, const SDF &sdf, const std::vector<std::array<Vec3, 3>> &triangles) {
    BVHTriangle bvh(triangles);
    Vec3 minAABB, maxAABB;
    bvh.getAABB(minAABB, maxAABB);

    Parallel::ForProgress(0, atoms.size(), [&sdf, &atoms, &bvh, layerHeight, minAABB](size_t i) {
        if (sdf.getVal(atoms[i]) > -0.1f*layerHeight) {
            atoms[i] = bvh.closestPoint(atoms[i]+minAABB)-minAABB;
        }
    });
}
