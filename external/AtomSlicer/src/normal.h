// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include "sdf.h"
#include "grid.h"
#include <string>

class NormalField {
public:
   enum Objective {
        ConformalSmooth = 0,
        ConformalTop = 1,
        SupportFree = 2,
        ConformalNearest = 3,
    };

    NormalField(const SDF &sdf, float layerHeight, float maxTopAngle = 30.0f, float maxBottomAngle = 1.0f, Objective objective = Objective::ConformalTop, uint32_t hardCodedField = 0, float numberOfCover = 4.0f);

    const std::vector<Vec3> &getNormals() const {
        return normals;
    }

    static std::string objectiveToString(Objective objective) {
        switch (objective) {
        case ConformalSmooth:
            return "Conformal, smoothly interpolated";
        case ConformalTop:
            return "Conformal on top and bottom surfaces, planar otherwise";
        case SupportFree:
            return "Support free";
        case ConformalNearest:
            return "Oriented as the closest conformal surface";
        default:
            return "";
        }
    }

private:
    struct Data {
        Vec3 normal{0,0,1};
        float distance{0};
        uint8_t flags{0};
    };
    enum Flags {
        Frozen = 0b01,
        Outside = 0b10,
    };
    enum Region {
        Bottom = 0b0000,
        Top = 0b0100,
        Wall = 0b1000,
        Other = 0b1100,
    };

    static void init(size_t i, const std::vector<float> &sdf, GridHierarchy<Data> &grid, float layerHeight, float maxTopAngle, float maxBottomAngle);
    static void restrict(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse);
    static void prolong(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse);
    static void smooth(size_t i, GridHierarchy<Data> *grid);
    static void smoothAll(size_t i, GridHierarchy<Data> *grid);
    static void smoothFrozen(size_t i, GridHierarchy<Data> *grid);

    static Vec3 gradSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid);
    static Vec3 gradDist(size_t i, const GridHierarchy<Data> &grid);
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

    std::vector<Vec3> normals;
};
