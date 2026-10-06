// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include "sdf.h"
#include "grid.h"
#include "normal.h"
#include <iostream>
#include <chrono>

class TriPhasor {
    struct Data {
        Vec3 normal{ 0,0,1 };
        Vec3 tangent{ 1,0,0 };
        float phase[3]{};
        uint8_t flags{};
    };
public:
    TriPhasor(const SDF &sdf, const SDF &originalSdf, const std::vector<Vec3> &normals, float layerHeight, float maxTopAngle = 30.0f, float maxBottomAngle = 1.0f, NormalField::Objective objective = NormalField::Objective::ConformalTop);

    const std::vector<Vec3> &getPoints() const { return atomPosition; }
    const std::vector<Vec3> &getNormals() const { return atomNormal; }

    void saveAtomsToPLY(const char *path, float layerHeight) const;

private:
    enum Flags {
        Frozen = 0b01,
        Outside = 0b10,
    };
    enum Region {
        Bottom = 0b0000,
        Top = 0b0100,
        Wall = 0b1000,
        Other = 0b1100
    };

    static void initLayers(size_t i, const std::vector<float> &sdf, const std::vector<Vec3> &normals, GridHierarchy<Data> &grid, float layerHeight, float maxTopAngle, float maxBottomAngle, NormalField::Objective objective);
    static void restrictLayers(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight);
    static void prolongLayers(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight);
    static void smoothLayers(size_t i, GridHierarchy<Data> *grid, float layerHeight);

    static void initTangents(size_t i, const std::vector<float> &sdf, const std::vector<Vec3> &normals, GridHierarchy<Data> &grid, float layerHeight, float maxTopAngle, float maxBottomAngle, NormalField::Objective objective);
    static void restrictTangents(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse);
    static void prolongTangents(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse);
    static void smoothTangents(size_t i, GridHierarchy<Data> *grid);
    static void smoothTangentsAll(size_t i, GridHierarchy<Data> *grid);

    static void restrictTriphasor(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight);
    static void prolongTriphasor(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight);
    static void smoothTriphasor(size_t i, GridHierarchy<Data> *grid, float layerHeight);

    void extractAtoms(const SDF &sdf, GridHierarchy<Data> &grid, float layerHeight);
    void saveSliceToPLY(const char *path, const GridHierarchy<Data> &grid, float layerHeight) const;

    static Vec3 gradSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid);
    static float curvatureSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid);

    static void setFrozen(uint8_t &flags, bool val) {
        flags = (flags & ~Frozen) | val;
    }

    static void setOutside(uint8_t &flags, bool val) {
        flags = (flags & ~Outside) | (val << 1);
    }

    static bool isFrozen(const uint8_t &flags) {
        return flags & Frozen;
    }

    static bool isOutside(const uint8_t &flags) {
        return flags & Outside;
    }

    static Region getRegion(const uint8_t &flags) {
        return Region(flags & 0b1100);
    }

    static void setRegion(uint8_t &flags, Region region) {
        flags = (flags & ~0b1100) | region;
    }

    std::vector<Vec3> atomPosition;
    std::vector<Vec3> atomNormal;
};