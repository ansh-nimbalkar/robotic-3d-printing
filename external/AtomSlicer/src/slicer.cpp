// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "slicer.h"
#include "sdf.h"
#include "normal.h"
#include "triphasor.h"
#include "partition.h"
#include "machine.h"
#include "meshing.h"
#include "snapper.h"

#include <chrono>
#include <iostream>

Slicer::Slicer(const std::vector<std::array<Vec3, 3>> &triangles, float nozzleWidth, float layerHeight, float maxTopAngle, float maxBottomAngle, Infill::Type infillType, NormalField::Objective objective, uint32_t hardCodedField, float numberOfCover, uint32_t zigzagOffset, float nozzleConeAngle, bool upCollisionCheck, bool holeClosing) : layerHeight(layerHeight) {
    auto startTime = std::chrono::steady_clock::now();
    std::cout << "\nSDF Generation..." << std::endl;
    SDF *sdf = new SDF(triangles, cellSizeFromLayerHeight(layerHeight), 2);
    std::cout << "SDF Size: [" << sdf->getSize(0) << ", " << sdf->getSize(1) << ", " << sdf->getSize(2) << "]" << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nOptimizing Tool Orientations..." << std::endl;
    if (hardCodedField == 1 /* Dragon */) {
        objective = NormalField::Objective::SupportFree;
    }
    NormalField *normalField = new NormalField(*sdf, layerHeight, maxTopAngle, maxBottomAngle, objective, hardCodedField, numberOfCover);
    auto normals = normalField->getNormals();
    delete normalField;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nGenerating Infills..." << std::endl;
    SDF *originalSDF = new SDF(*sdf);
    Infill::generate(*sdf, infillType, nozzleWidth);
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    TriPhasor *triPhasor = new TriPhasor(*sdf, *originalSDF, normals, layerHeight, maxTopAngle, maxBottomAngle, objective);
    auto points = triPhasor->getPoints();
    normals = triPhasor->getNormals();
    delete triPhasor;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nSnapping..." << std::endl;
    Snapper::snap(points, layerHeight, *originalSDF, triangles);
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nPartitioning..." << std::endl;
    Partition partition(points, normals, layerHeight, nozzleConeAngle, upCollisionCheck);

    std::cout << "Layers: " << partition.getLayers().size() << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nMeshing..." << std::endl;
    Meshing meshing(points, normals, partition.getLayers(), nozzleWidth, layerHeight, *sdf);
    delete sdf;
    //partition.savePartitionToPLY("../data/test.ply");
    //meshing.saveLayersToPLY("../data/layers/");
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nToolpath generation..." << std::endl;
    toolpath = Toolpath(meshing.getLayers(), *originalSDF, points, normals, nozzleWidth, layerHeight, hardCodedField, zigzagOffset, holeClosing, false);
    toolpath.smoothNormals(128);
    toolpath.smoothPositions(2, *originalSDF, nozzleWidth, holeClosing);
    delete originalSDF;
    //toolpath.saveToPLY("../data/test.ply", layerHeight);
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;
}

void Slicer::saveToGCode(const char *path) const {
    auto startTime = std::chrono::steady_clock::now();
    std::cout << "\nGCode generation..." << std::endl;
    Toolpath toolpathCopy(toolpath);
    toolpathCopy.tessellateNormals(1.0);
    Machine::addPlatformAndCenter(toolpathCopy, 2.0f*layerHeight, layerHeight);
    Machine::toolpathToGCode(toolpathCopy, path, 2.0f*layerHeight, layerHeight);
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;
}

void Slicer::saveToPLY(const char *path) const {
    toolpath.saveToPLY(path, layerHeight);
}

void Slicer::saveToGCode3Axis(const char *path) const {
    auto startTime = std::chrono::steady_clock::now();
    std::cout << "\n3-Axis GCode generation..." << std::endl;
    Toolpath toolpathCopy(toolpath);
    Machine::center3Axis(toolpathCopy, layerHeight);
    Machine::toolpathToGCode3Axis(toolpathCopy, path, 2.0f*layerHeight, layerHeight);
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;
}