// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "curve.h"
#include "bvhsegment.h"
#include <chrono>
#include <iostream>

Curve::Curve(const float *vertsPositions, uint32_t vertsCount, const uint32_t *triangleIndices, uint32_t triangleCount,
    float width, const SDF &sdf, const float *directions, Stripe::Mode mode, bool resample, bool holeClosing, bool quiet) : width (width), quiet(quiet) {
    auto startTotalTime = std::chrono::steady_clock::now();
    stripe = Stripe(vertsPositions, directions, triangleIndices, vertsCount, triangleCount, width, mode, quiet);
    auto startTime = std::chrono::steady_clock::now();
    stitcher = Stitcher(stripe.getPositions(), stripe.getNormals(), stripe.getTriangle(), stripe.getVertCount(), stripe.getTriangleCount(), stripe.getBorderEdges(), sdf);
    if (!quiet) std::cout << "Stitcher initialization (" << stripe.getVertCount() << " Verts): " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count() << " ms" << std::endl;

    startTime = std::chrono::steady_clock::now();
    stripe.optimize();
    if (!quiet) std::cout << "Stripes generation: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count() << " ms" << std::endl;

    startTime = std::chrono::steady_clock::now();
    stitcher.stitch(stripe.getScalars(), stripe.getPeriod(), holeClosing, !quiet);
    if (!quiet) std::cout << "Stitching: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count() << " ms" << std::endl;

    startTime = std::chrono::steady_clock::now();
    if (!holeClosing) {
        stitcher.repulse();
    }
    if (!quiet) std::cout << "Repulse: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count() << " ms" << std::endl;

    startTime = std::chrono::steady_clock::now();
    isolines = Isolines(stitcher.getPositions(), stitcher.getNormals(), stitcher.getTriangle(), stitcher.getVertCount(), stitcher.getTriangleCount());
    isolines.extract(stitcher.getScalars());
    if (resample) {
        isolines.resample(0.5f*width);
    }
    cycles = isolines.getIsolines();
    std::vector<std::array<Vec3, 2>> border;
    for (const auto &edge : stitcher.getBorderEdges()) {
        border.push_back({ Vec3(stitcher.getPositions(), edge[0]), Vec3(stitcher.getPositions(), edge[1]) });
    }
    snapCycles(sdf, border);
    uint32_t verts = 0;
    for (auto &c : cycles) verts += c.size();
    if (!quiet) std::cout << "Isolines (" << cycles.size() << " cycle, " << verts << " verts): "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count() << " ms" << std::endl;
    if (!quiet) std::cout << "==============\nTotal time (w/o IO): "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTotalTime).count() << " ms" << std::endl;
}

void Curve::snapCycles(const SDF &sdf, const std::vector<std::array<Vec3, 2>> &border) {
    if (!border.size()) return;
    BVHSegment bvh(border);
    for (auto &cycle : cycles) {
        Parallel::For(0, cycle.size(), [this, &sdf, &cycle, &bvh](size_t i) {
            Vec3 pos = cycle[i][0];
            float sdfVal = sdf.getVal(pos);
            float targetDist = width*0.5f;
            float margin = 0.3f*width;
            if (fabsf(sdfVal + targetDist) < margin) {
                Vec3 closest = bvh.closestPoint(pos);
                Vec3 dir = pos-closest;
                if (fabsf(dir.length()-targetDist) < margin) {
                    Vec3 newPos = closest + dir.normalize() * targetDist;
                    cycle[i][0] = newPos;
                }
            }
        });
    }
}

const std::vector<std::vector<std::array<Vec3, 2>>> &Curve::getCycles() const {
    return cycles;
}

void Curve::saveThickDirectionsToPLY(const char *path) const {
    stripe.saveDirectionsToPLY(path, 0.03f*stripe.getPeriod(), 0.2f*stripe.getPeriod());
}

void Curve::saveScalarsToPLY(const char *path) const {
    stitcher.saveMeshToPLY(path);
}

void Curve::saveCurveToPLY(const char *path) const {
    isolines.saveToPLY(path, 0.02f*stripe.getPeriod());
}

void Curve::saveThickCurveToPLY(const char *path) const {
    isolines.saveToPLY(path, 0.02f*stripe.getPeriod(), true);
}

void Curve::saveDirectionsToPLY(const char *path) const {
    stripe.saveMeshToPLY(path, true);
}

void Curve::saveCutsToPLY(const char *path) const {
    stripe.saveCutsToPLY(path);
}

void Curve::saveMeshAndCurveToOBJ(const char *path) const {
    isolines.saveMeshAndCurveToOBJ(path);
}

void Curve::saveStripeToPLY(const char* path) const {
    stripe.saveMeshToPLY(path);
}
