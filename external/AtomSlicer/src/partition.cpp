// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "partition.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

void Partition::savePoints(const std::unordered_set<size_t> &layer, const std::string &path) const {
    std::ofstream plyFile(path);
    plyFile << "ply\nformat ascii 1.0\nelement vertex " << layer.size() << "\n"
            << "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
            << "end_header\n";
        
    for (auto i : layer) {
        plyFile << points[i][0] << " " << points[i][1] << " " << points[i][2] << std::endl;
    }
}

bool Partition::isNotLocallyBlocked(size_t i) const {
    const float cosHalfAngle = cosf(0.5f*nozzleConeAngle);
    for (auto j : bvh->pointsInSphereIndex(points[i], 2.0f*layerHeight)) {
        if (i == j) continue;
        Vec3 v = points[j]-points[i];
        float length = v.length();
        Vec3 n = !upCollisionCheck ? normals[i] : Vec3(0, 0, 1);
        float cosAngle = v.dot(n)/length;
        if (length <= 2.0f*layerHeight && cosAngle >= cosHalfAngle) {
            return false;
        }
    }
    return true;
}

bool Partition::isNeighboor(size_t i, size_t j) const {
    Vec3 v = points[j]-points[i];
    return fabsf(v.dot(normals[i])) < 0.5f*layerHeight && fabsf(v.dot(normals[j])) < 0.5f*layerHeight && v.length() <= 1.5f*layerHeight;
}

size_t Partition::getBlocker(size_t i) const {
    Vec3 n = !upCollisionCheck ? normals[i] : Vec3(0, 0, 1);
    return bvh->anyPointsInCone(points[i]+n*1e-3, n, nozzleConeAngle * 0.5);
}

void Partition::flood(size_t start, std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked) const {
    layer.clear();
    blocked.clear();
    std::vector<size_t> stack;

    layer.insert(start);
    stack.push_back(start);
    while (stack.size()) {
        size_t i = stack.back();
        stack.pop_back();
        for (auto j : bvh->pointsInSphereIndex(points[i], 1.5f*layerHeight)) {
            if (!layer.count(j) && !blocked.count(j) && isNeighboor(i, j)) {
                if (getBlocker(j) == size_t(-1)) {
                    layer.insert(j);
                    stack.push_back(j);
                } else {
                    blocked.insert(j);
                }
            }
        }
    }
}

void Partition::getBlockers(std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked, std::unordered_set<size_t> &blocker) const {
    blocker.clear();
    for (auto blockedAtom : blocked) {
        size_t blockerAtom = getBlocker(blockedAtom);
        size_t attempted = 0;
        while (!blocker.count(blockerAtom) && !blocked.count(blockerAtom) && !layer.count(blockerAtom)) {
            size_t nextBlocker = getBlocker(blockerAtom);
            if (attempted++ > points.size()) {
                std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
                exit(1);
            }
            if (nextBlocker == size_t(-1)) {
                blocker.insert(blockerAtom);
                break;
            }
            blockerAtom = nextBlocker;
        }
    }
}

size_t Partition::getHigherBlocker(std::unordered_set<size_t> &blocker) const {
    size_t best = *blocker.begin();
    for (auto atom : blocker) {
        if (points[atom][2] > points[best][2]) {
            best = atom;
        }
    }
    return best;
}

void Partition::saveLayersPLY(const char *path) const {
    for (size_t i = 0; i < layers.size(); ++i) {
        std::ofstream plyFile(std::string(path)+ std::to_string(i) +".ply");
        plyFile << "ply\nformat ascii 1.0\nelement vertex " << layers[i].size() << "\n"
                << "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
                << "end_header\n";
            
        for (auto j : layers[i]) {
            plyFile << points[j][0] << " " << points[j][1] << " " << points[j][2] << std::endl;
        }
    }
}

void Partition::savePartitionToPLY(const char *path) const {
    std::vector<Vec3> pos;
    std::vector<uint32_t> idx;
    std::vector<std::array<uint8_t, 3>> col;

    for (size_t i = 0; i < layers.size(); ++i) {            
        for (auto j : layers[i]) {
            pos.push_back(points[j]);
            idx.push_back(i);
            col.push_back(turbo(float(i)/(layers.size()-1)));
        }
    }

    std::ofstream plyFile(path);   
    plyFile << "ply\nformat ascii 1.0\nelement vertex " << pos.size() << "\n"
            << "property float x\nproperty float y\nproperty float z\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            << "property uint order\n"
            << "end_header\n";
    for (size_t i = 0; i < pos.size(); ++i) {
        plyFile << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << " "
                << uint32_t(col[i][0]) << " " << uint32_t(col[i][1]) << " " << uint32_t(col[i][2]) << " "
                << idx[i] << std::endl;
    }
}

void Partition::saveLayersIndexCSV(const char *path) const {
    std::ofstream layersFile(path);
    for (auto &layer : layers) {
        for (auto i : layer) {
            layersFile << i << std::endl;
        }
        layersFile << "===" << std::endl;
    }
}

Partition::Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const char *indexCSVPath) : points(points), normals(normals) {
    std::ifstream file(indexCSVPath);
    std::string line;
    size_t idx;
    std::vector<size_t> layer;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> idx)) {
            layers.push_back(layer);
            layer.clear();
            continue;
        }
        layer.emplace_back(idx);
    }
}

void Partition::saveLayersCSV(const char *path) const {
    std::ofstream layersFile(path);
    for (auto &layer : layers) {
        for (auto i : layer) {
            layersFile << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << std::endl;
        }
        layersFile << "===" << std::endl;
    }
}

Partition::Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float layerHeight, float nozzleConeAngle, bool upCollisionCheck, bool quiet) : layerHeight(layerHeight), points(points), normals(normals), bvh(new BVHPoint(points)), nozzleConeAngle(nozzleConeAngle/180.0f*M_PI), upCollisionCheck(upCollisionCheck) {
    if (!points.size()) return;
    std::unordered_set<size_t> layer;
    std::unordered_set<size_t> blocked;
    std::unordered_set<size_t> blocker;
    std::unordered_set<size_t> locallyViables;
    std::vector<std::unordered_set<size_t>> attemptedLayers;
    size_t added = 0;
    size_t backtracks = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        if (isNotLocallyBlocked(i)) {
            locallyViables.insert(i);
        }
    }

    if (!locallyViables.size()) {
        std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
        exit(1);
    }

    // Start from the higher atom
    size_t startPoint = *locallyViables.begin();
    for (auto i : locallyViables) {
        if (points[i][2] > points[startPoint][2]) {
            if (getBlocker(i) == size_t(-1)) {
                startPoint = i;
            }
        }
    }

    while (locallyViables.size()) {
        // Flood the layer and keep track of the non viable neighbours
        flood(startPoint, layer, blocked);

        // Get the blockers that are not in the layer
        getBlockers(layer, blocked, blocker);

        startPoint = size_t(-1);
        if (blocker.size()) {
            bool loop = false;
            for (auto &attempt : attemptedLayers) {
                if (attempt == layer) {
                    loop = true;
                    break;
                }
            }

            if (loop) {
                // If there is a cycle check if there is a fully accessible layer
                std::unordered_set<size_t> starts(locallyViables);
                std::unordered_set<size_t> layerTMP;
                for (size_t atom : layer) {
                    starts.erase(atom);
                }
                for (const auto &attempted : attemptedLayers) {
                    for (size_t atom : attempted) {
                        starts.erase(atom);
                    }
                }
                
                while (starts.size()) {
                    size_t startPoint = *starts.begin();
                    if (getBlocker(startPoint) != size_t(-1)) {
                        starts.erase(startPoint);
                        continue;
                    }
                    
                    flood(startPoint, layerTMP, blocked);
                    getBlockers(layerTMP, blocked, blocker);
                    // If there is, add that one
                    if (!blocker.size()) {
                        layer = std::move(layerTMP);
                        break;
                    }

                    for (size_t atom : layerTMP) {
                        starts.erase(atom);
                    }
                }
                

                // Add the current layer if there are not fully accessible ones
                layers.emplace_back(std::vector<size_t>(layer.begin(), layer.end()));
                attemptedLayers.clear();
            } else {
                // Start from the higher blocker
                startPoint = getHigherBlocker(blocker);
                attemptedLayers.push_back(layer);
            }
        } else {
            layers.emplace_back(std::vector<size_t>(layer.begin(), layer.end()));
            attemptedLayers.clear();
        }

        if (startPoint == size_t(-1)) {
            // Update the BVH and the locally viables
            for (auto i : layer) {
                bvh->removePoint(i);
                locallyViables.erase(i);
            }
            for (auto i : layer) {
                for (auto j : bvh->pointsInSphereIndex(points[i], 2.0f*layerHeight + 1e-3)) {
                    if (!locallyViables.count(j) && isNotLocallyBlocked(j)) {
                        locallyViables.insert(j);
                    }
                }
            }

            // Continue from the closest to the first atom of the last added layer
            float closestDist = std::numeric_limits<float>::infinity();
            bool found = false;
            for (auto i : locallyViables) {
                float distance = (points[*layer.begin()]-points[i]).length2();
                if (distance < closestDist) {
                    if (getBlocker(i) == size_t(-1)) {
                        closestDist = distance;
                        startPoint = i;
                        found = true;
                    }
                }
            }
            if (!found && locallyViables.size()) {
                std::cout << "\nFatal Error: bidirectional constraint! Try reducing the maximum tilting angles." << std::endl;
                exit(1);
            }

            added += layer.size();
            printProgress(float(added)/points.size());
        } else {
            backtracks++;
        }
    }
    assert(added == points.size());

    // Filter too small layers
    layers.erase(
        std::remove_if(layers.begin(), layers.end(),
            [](const auto& layer) { return layer.size() <= 8; }),
        layers.end()
    );

    std::reverse(layers.begin(), layers.end());
    delete bvh;
    bvh = nullptr;

    if (!quiet) std::cout << "Backtracks: " << backtracks << std::endl;
}