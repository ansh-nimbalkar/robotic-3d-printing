// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#pragma once

#include "bvhpoint.h"
#include "machine.h"
#include <cassert>
#include <string>

class Partition {
public:
    Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float layerHeight, float nozzleConeAngle = 95.0f, bool upCollisionCheck = false, bool quiet = false);
    Partition(const std::vector<Vec3> &points, const std::vector<Vec3> &normals, const char *indexCSVPath);
    const std::vector<std::vector<size_t>> &getLayers() const { return layers; }

    void saveLayersCSV(const char *path) const;
    void saveLayersIndexCSV(const char *path) const;
    void saveLayersPLY(const char *path) const;
    void savePartitionToPLY(const char *path) const;

private:
    bool isNotLocallyBlocked(size_t i) const;
    bool isNeighboor(size_t i, size_t j) const;
    size_t getBlocker(size_t i) const;
    void flood(size_t start, std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked) const;
    void getBlockers(std::unordered_set<size_t> &layer, std::unordered_set<size_t> &blocked, std::unordered_set<size_t> &blocker) const;
    size_t getHigherBlocker(std::unordered_set<size_t> &blocker) const;
    void savePoints(const std::unordered_set<size_t> &layer, const std::string &path) const;

    float layerHeight;
    std::vector<std::vector<size_t>> layers;
    const std::vector<Vec3> &points;
    const std::vector<Vec3> &normals;
    BVHPoint *bvh;
    float nozzleConeAngle;
    bool upCollisionCheck;
};