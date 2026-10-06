// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "meshing.h"
#include "parallel.h"
#include <fstream>
#include <string>
#include <deque>
#include <algorithm>

Meshing::Meshing(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const std::vector<std::vector<size_t>> &partition, float width, float height, const SDF &sdf) {
    layers.resize(partition.size());
    Parallel::ForProgress(0, layers.size(), [&](size_t i) {
        layers[i] = Meshing::mesh(partition[i], points, normals, width, height, sdf);
    });
}

void Meshing::saveLayersToPLY(const char *path) const {
    for (size_t i = 0; i < layers.size(); ++i) {
        std::ofstream plyFile(std::string(path)+ std::to_string(i) +".ply");
        plyFile << "ply\nformat ascii 1.0\nelement vertex " << layers[i].vertices.size()/3 << "\n"
                << "property float x\nproperty float y\nproperty float z\n"
                << "element face " << layers[i].triangles.size()/3 << "\nproperty list uchar uint vertex_indices\n"
                << "end_header\n";

        for (size_t j = 0; j < layers[i].vertices.size()/3; ++j) {
            plyFile << layers[i].vertices[3*j+0] << " " << layers[i].vertices[3*j+1] << " " << layers[i].vertices[3*j+2] << std::endl;
        }

        for (size_t j = 0; j < layers[i].triangles.size()/3; ++j) {
            plyFile << "3 " << layers[i].triangles[3*j+0] << " " << layers[i].triangles[3*j+1] << " " << layers[i].triangles[3*j+2] << std::endl;
        }
    }
}

Meshing::Tris Meshing::getSeedTriangle(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const BVHPoint &bvh, float layerHeight, const std::unordered_set<size_t> &usedVertices, const SDF &sdf) {
    for (size_t a = 0; a < points.size(); ++a) {
        if (usedVertices.count(a)) continue;
        auto neighboors = bvh.pointsInSphereIndex(points[a], 2.0f*layerHeight);
        float distance = std::numeric_limits<float>::infinity();
        size_t b = 0;
        for (const auto &n : neighboors) {
            if (n == a) continue;
            float dist = (points[n]-points[a]).length2();
            if (dist < distance) {
                distance = dist;
                b = n;
            }
        }
        if (std::isinf(distance)) continue;

        for (const auto &c : neighboors) {
            if (c == a || c == b) continue;
            if (sdf.getVal((points[a]+points[b]+points[c])/3.0f) > 0.2f*layerHeight) continue;

            Vec3 v0 = points[a]-points[c], v1 = points[b]-points[c];
            if (fabsf(v0.normalize().dot(v1.normalize())) > 0.9) continue;

            float radius = 0.5f*v0.length()*v1.length()*(v0-v1).length()/v0.cross(v1).length();
            Vec3 center = 0.5f*(v0.length2()*v1 - v1.length2()*v0).cross(v0.cross(v1))/v0.cross(v1).length2() + points[c];

            bool valid = true;
            for (const auto &t : bvh.pointsInSphereIndex(center, radius+1e-2f*layerHeight)) {
                if (t == a || t == b || t == c) continue;
                if ((points[t]-center).length() < radius - 1e-2*layerHeight) valid = false;
            }
            if (valid) {
                Vec3 norm = (points[c]-points[a]).cross((points[b]-points[a])).normalize();
                Vec3 avgNorm = (normals[c]+normals[a]+normals[b]).normalize();
                if (norm.dot(avgNorm) < 0.0) {
                    return Tris{a, b, c};
                } else {
                    return Tris{b, a, c};
                }
            }
        }
    }
    return Tris{size_t(-1), size_t(-1), size_t(-1)};
}

Meshing::Tris Meshing::getNextTriangle(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const Edge &edge, const BVHPoint &bvh, float layerHeight, const std::unordered_set<Edge> &usedEdges, const SDF &sdf) {
    Vec3 midPoint = 0.5f*(points[edge.a]+points[edge.b]);
    auto neighboors = bvh.pointsInSphereIndex(midPoint, 2.0f*layerHeight);
    std::sort(neighboors.begin(), neighboors.end(), [&](size_t a, size_t b) {
        return (points[a]-midPoint).length2() < (points[b]-midPoint).length2();
    });
    
    for (const auto &c : neighboors) {
        if (c == edge.a || c == edge.b) continue;
        if (sdf.getVal((points[edge.a]+points[edge.b]+points[c])/3.0f) > 0.2f*layerHeight) continue;

        if (usedEdges.count(Edge(c, edge.b))) continue;
        if (usedEdges.count(Edge(edge.b, edge.a))) continue;
        if (usedEdges.count(Edge(edge.a, c))) continue;

        Vec3 v0 = points[edge.a]-points[c], v1 = points[edge.b]-points[c];
        if (fabsf(v0.normalize().dot(v1.normalize())) > 0.9) continue;

        Vec3 norm = (points[c]-points[edge.a]).cross((points[edge.b]-points[edge.a])).normalize();
        Vec3 avgNorm = (normals[c]+normals[edge.a]+normals[edge.b]).normalize();
        if (norm.dot(avgNorm) < 0.2) continue;

        float radius = 0.5f*v0.length()*v1.length()*(v0-v1).length()/v0.cross(v1).length();
        Vec3 center = 0.5f*(v0.length2()*v1 - v1.length2()*v0).cross(v0.cross(v1))/v0.cross(v1).length2() + points[c];

        bool valid = true;
        for (const auto &t : bvh.pointsInSphereIndex(center, radius+1e-2f*layerHeight)) {
            if (t == edge.a || t == edge.b || t == c) continue;
            if ((points[t]-center).length() < radius - 1e-2*layerHeight) valid = false;
        }
        if (valid) {
            return Tris{c, edge.b, edge.a};
        }
    }
    
    return Tris{size_t(-1), size_t(-1), size_t(-1)};
}

Toolpath::Mesh Meshing::mesh(const std::vector<size_t> &layer, const std::vector<Vec3> &allPoints, const std::vector<Vec3> &allNormals, float width, float layerHeight, const SDF &sdf) {
    std::vector<Vec3> points;
    std::vector<Vec3> normals;
    for (size_t i : layer) {
        points.push_back(allPoints[i]);
        normals.push_back(allNormals[i]);
    }

    std::vector<Tris> triangles;
    std::vector<Edge> borders;
    BVHPoint bvh(points);
    std::unordered_set<size_t> usedVertices;

    while (true) {
        Tris seed = getSeedTriangle(points, normals, bvh, layerHeight, usedVertices, sdf);
        if (seed.a == size_t(-1)) {
            break;
        }

        std::unordered_set<Edge> usedEdges;
        std::deque<Edge> frontier;
        auto addTriangle = [&](const Tris &tris) {
            triangles.push_back(tris);
            if (!usedEdges.count(Edge(tris.a, tris.b))) frontier.push_back(Edge(tris.a, tris.b));
            if (!usedEdges.count(Edge(tris.b, tris.c))) frontier.push_back(Edge(tris.b, tris.c));
            if (!usedEdges.count(Edge(tris.c, tris.a))) frontier.push_back(Edge(tris.c, tris.a));
            usedEdges.insert(Edge(tris.a, tris.b));
            usedEdges.insert(Edge(tris.b, tris.c));
            usedEdges.insert(Edge(tris.c, tris.a));
        };
        addTriangle(seed);

        while (frontier.size()) {
            Edge edge = frontier.front(); frontier.pop_front();
            Tris next = getNextTriangle(points, normals, edge, bvh, layerHeight, usedEdges, sdf);
            if (next.a != size_t(-1)) {
                addTriangle(next);
            }
        }

        // Close holes
        std::unordered_set<Edge> borderEdges;
        for (const auto &edge : usedEdges) {
            if (!usedEdges.count({edge.b, edge.a})) {
                borderEdges.insert(edge);
            }
        }
        std::vector<std::vector<size_t>> perimeters;
        std::vector<size_t> perimeter;
        while (borderEdges.size()) {
            perimeter.clear();
            perimeter.push_back(borderEdges.begin()->a);
            perimeter.push_back(borderEdges.begin()->b);
            borderEdges.erase(borderEdges.begin());

            bool found = true;
            while(found) {
                found = false;
                for (const Edge &e : borderEdges) {
                    if (e.a == perimeter.back()) {
                        perimeter.push_back(e.b);
                        borderEdges.erase(e);
                        found = true;
                        break;
                    }

                    if (e.b == perimeter.back()) {
                        perimeter.push_back(e.a);
                        borderEdges.erase(e);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    perimeters.push_back(perimeter);
                }
            }
        }

        for (const auto &p : perimeters) {
            if (p.size() == 4 && p[0] == p[3]) {
                size_t a = p[0], b = p[1], c = p[2];
                Vec3 norm = (points[c]-points[a]).cross((points[b]-points[a])).normalize();
                Vec3 avgNorm = (normals[c]+normals[a]+normals[b]).normalize();
                Tris t = norm.dot(avgNorm) < 0.0 ? Tris{a, b, c} : Tris{b, a, c};
                bool found = false;
                for (const auto &o : triangles) {
                    if ((t.a == o.a && t.b == o.b && t.c == o.c) ||
                        (t.a == o.b && t.b == o.c && t.c == o.a) ||
                        (t.a == o.c && t.b == o.a && t.c == o.b)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    triangles.push_back(t);
                }
            }
        }

        for (auto &tris : triangles) {
            usedVertices.insert(tris.a);
            usedVertices.insert(tris.b);
            usedVertices.insert(tris.c);
            bvh.removePoint(tris.a);
            bvh.removePoint(tris.b);
            bvh.removePoint(tris.c);
        }
    }

    // Removed isolated vertices
    std::unordered_map<size_t, size_t> indexMap;
    std::vector<Vec3> usedPoints;
    usedPoints.reserve(triangles.size() * 3);
    for (const auto &f : triangles) {
        for (uint8_t i = 0; i < 3; ++i) {
            size_t oldIdx = f[i];
            if (indexMap.find(oldIdx) == indexMap.end()) {
                indexMap[oldIdx] = usedPoints.size();
                usedPoints.push_back(points[oldIdx]);
            }
        }
    }

    // Convert to Toolpath::Mesh format
    std::vector<float> verts(3*usedPoints.size());
    for (size_t i = 0; i < usedPoints.size(); ++i) {
        verts[3*i+0] = usedPoints[i].x;
        verts[3*i+1] = usedPoints[i].y;
        verts[3*i+2] = usedPoints[i].z;
    }

    std::vector<uint32_t> tris(3*triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        tris[3*i+0] = indexMap[triangles[i].a];
        tris[3*i+1] = indexMap[triangles[i].b];
        tris[3*i+2] = indexMap[triangles[i].c];
    }

    return {std::move(verts), std::move(tris)};
}
