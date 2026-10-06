// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once
#include "toolpath.h"
#include "sdf.h"
#include "bvhpoint.h"

class Meshing {
public:
    Meshing(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const std::vector<std::vector<size_t>> &partition, float width, float height, const SDF &sdf);

    const std::vector<Toolpath::Mesh> &getLayers() const { return layers; }
    void saveLayersToPLY(const char *path) const;

    struct Tris {
        size_t a, b, c;
        const size_t &operator[](size_t i) const { if (i == 0) return a; if (i == 1) return b; return c; }
    };

    struct Edge {
        Edge(size_t v0, size_t v1) : a(v0), b(v1) {}
        bool operator==(const Edge &o) const { return a == o.a && b == o.b; }
        size_t a, b;
    };

private:
    static Toolpath::Mesh mesh(const std::vector<size_t> &layer, const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float width, float height, const SDF &sdf);
    static Tris getSeedTriangle(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const BVHPoint &bvh, float layerHeight, const std::unordered_set<size_t> &usedVertices, const SDF &sdf);
    static Tris getNextTriangle(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const Edge &edge, const BVHPoint &bvh, float layerHeight, const std::unordered_set<Edge> &usedEdges, const SDF &sdf);

    std::vector<Toolpath::Mesh> layers;
};

template <> 
struct std::hash<Meshing::Edge> {
    std::size_t operator()(const Meshing::Edge& e) const {
    return (std::hash<size_t>()(e.a) ^ (std::hash<size_t>()(e.b) << 1)) >> 1;
    }
};