// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "triphasor.h"

TriPhasor::TriPhasor(const SDF &sdf, const SDF &originalSdf, const std::vector<Vec3> &normals, float layerHeight, float maxTopAngle, float maxBottomAngle, NormalField::Objective objective) {
    GridHierarchy<Data> grid({sdf.getSize(0), sdf.getSize(1), sdf.getSize(2)}, sdf.getCellSize(), Vec3(-2*sdf.getCellSize()));

    auto startTime = std::chrono::steady_clock::now();
    std::cout << "\nImplicit Layers Generation..." << std::endl;
    Parallel::For(0, grid.getTotalSize(), [this, &originalSdf, &grid, &normals, layerHeight, maxTopAngle, maxBottomAngle, objective](size_t i) {
        initLayers(i, originalSdf.getField(), normals, grid, layerHeight, maxTopAngle, maxBottomAngle, objective);
    });
    auto restrict = [this, layerHeight](size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) { restrictLayers(i, gridDense, gridCoarse, layerHeight); };
    auto smooth = [this, layerHeight](size_t i, GridHierarchy<Data> *grid) { smoothLayers(i, grid, layerHeight); };
    auto prolong = [this, layerHeight](size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) { prolongLayers(i, gridDense, gridCoarse, layerHeight); };
    grid.optimize(smooth, restrict, prolong);

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nOptimizing Tangents..." << std::endl;
    Parallel::For(0, grid.getTotalSize(), [this, &sdf, &grid, &normals, layerHeight, maxTopAngle, maxBottomAngle, objective](size_t i) {
        initTangents(i, sdf.getField(), normals, grid, layerHeight, maxTopAngle, maxBottomAngle, objective);
    });
    grid.optimize(smoothTangents, restrictTangents, prolongTangents);
    grid.smooth(smoothTangentsAll, 4);

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nOptimizing Triphasor..." << std::endl;
    Parallel::For(0, grid.getTotalSize(), [this, &sdf, &grid, &normals, layerHeight, maxTopAngle, maxBottomAngle](size_t i) {
        Vec3 grad = gradSDF(i, sdf.getField(), grid);
        Vec3 bitangent = grid.getData(i)->normal.cross(grid.getData(i)->tangent).normalize();
        grid.getData(i)->phase[1] = 2.0f*M_PI*(sdf.getField()[i]/layerHeight) * (grad.dot(bitangent) < 0 ? -1 : 1);
    });
    auto restrictTri = [this, layerHeight](size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) { restrictTriphasor(i, gridDense, gridCoarse, layerHeight); };
    auto smoothTri = [this, layerHeight](size_t i, GridHierarchy<Data> *grid) { smoothTriphasor(i, grid, layerHeight); };
    auto prolongTri = [this, layerHeight](size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) { prolongTriphasor(i, gridDense, gridCoarse, layerHeight); };
    grid.optimize(smoothTri, restrictTri, prolongTri);

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    startTime = std::chrono::steady_clock::now();
    std::cout << "\nExtracting Atoms..." << std::endl;
    extractAtoms(sdf, grid, layerHeight);
    std::cout << "Atoms count: " << atomPosition.size() << std::endl;

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    /*Parallel::For(0, grid.getTotalSize(), [this, &grid](size_t i) {
        setFrozen(grid.getData(i)->flags, false);
        setOutside(grid.getData(i)->flags, false);
    });
    grid.saveColorToPLY("../../data/field.ply", [&sdf](size_t i, const GridHierarchy<Data>* grid, bool &save) {
        save = sdf.getField()[sdf.getIndex(grid->getPosition(i))] < grid->getCellSize();
        float x = 0.5f * cosf(grid->getData(i)->phase[0]) + 0.5f;
        float y = 0.5f * cosf(grid->getData(i)->phase[1]) + 0.5f;
        float z = 0.5f * cosf(grid->getData(i)->phase[2]) + 0.5f;
        return viridis(z);
        return viridis((x+y+z)/3);
    }, [this, layerHeight](size_t i, GridHierarchy<Data>* gridDense, GridHierarchy<Data>* gridCoarse) { 
        prolongLayers(i, gridDense, gridCoarse, layerHeight);
        prolongTriphasor(i, gridDense, gridCoarse, layerHeight);
    }, 2);

    saveAtomsToPLY("../../data/atoms.ply", layerHeight);

    grid.saveFloatToBin("../../data/triphasor.bin", [&sdf](size_t i, const GridHierarchy<Data> *grid) {
        if (sdf.getField()[i] > 0) return 100.0f;
        float x = 0.5f*cosf(grid->getData(i)->phase[0])+0.5f;
        float y = 0.5f*cosf(grid->getData(i)->phase[1])+0.5f;
        float z = 0.5f*cosf(grid->getData(i)->phase[2])+0.5f;
        return (x+y+z)/3.0f;
    });

    grid.saveVec3ToPLY("../../data/test.ply", [](size_t i, const GridHierarchy<Data> *grid, bool &save) {
        save = getRegion(grid->getData(i)->flags) == Region::Bottom;
        return Vec3(0.1);
    });
    
    saveSliceToPLY("../../data/test.ply", grid, layerHeight);*/
}

void TriPhasor::extractAtoms(const SDF &sdf, GridHierarchy<Data> &grid, float layerHeight) {
    std::vector<float> maxVal(grid.getTotalSize(), -std::numeric_limits<float>::infinity());
    std::vector<Vec3> maxPos(grid.getTotalSize(), Vec3(std::numeric_limits<float>::quiet_NaN()));
    Parallel::ForProgress(0, grid.getTotalSize(), [&grid, &maxVal, &maxPos, layerHeight](size_t i) {
        if (isOutside(grid.getData(i)->flags)) return;
        Vec3 dirT = grid.getData(i)->tangent;
        Vec3 dirN = grid.getData(i)->normal;
        Vec3 dirB = dirN.cross(dirT).normalize();
        float phaseT = grid.getData(i)->phase[0];
        float phaseB = grid.getData(i)->phase[1];
        float phaseN = grid.getData(i)->phase[2];

        const uint8_t N = 8;
        float range = cellSizeFromLayerHeight(layerHeight) * (0.5f-0.5f/N);
        for (uint8_t x = 0; x < N; ++x) {
            for (uint8_t y = 0; y < N; ++y) {
                for (uint8_t z = 0; z < N; ++z) {
                    Vec3 pos = range * Vec3(2*(float(x)/(N-1))-1, 2*(float(y)/(N-1))-1, 2*(float(z)/(N-1))-1);
                    float valT = cosf(2.0*M_PI/layerHeight*dirT.dot(pos)+phaseT);
                    float valB = cosf(2.0*M_PI/layerHeight*dirB.dot(pos)+phaseB);
                    float valN = cosf(2.0*M_PI/layerHeight*dirN.dot(pos)+phaseN);

                    if (valT+valB+valN > maxVal[i]) {
                        maxVal[i] = valT+valB+valN;
                        maxPos[i] = grid.getPosition(i) + pos;
                    }
                }
            }
        }
    });

    Parallel::For(0, grid.getTotalSize(), [&grid, &maxVal, &maxPos, &sdf, layerHeight](size_t i) {
        if (isOutside(grid.getData(i)->flags)) return;
        if (sdf.getVal(maxPos[i]) >= 0.15f*layerHeight) {
            maxPos[i].x = std::numeric_limits<float>::quiet_NaN();
            return;
        }

        auto xyz = grid.idx1To3(i);
        uint32_t minX = xyz[0] > 0 ? xyz[0]-1 : xyz[0];
        uint32_t maxX = xyz[0] < grid.getSize()[0]-1 ? xyz[0]+1 : xyz[0];
        uint32_t minY = xyz[1] > 0 ? xyz[1]-1 : xyz[1];
        uint32_t maxY = xyz[1] < grid.getSize()[1]-1 ? xyz[1]+1 : xyz[1];
        uint32_t minZ = xyz[2] > 0 ? xyz[2]-1 : xyz[2];
        uint32_t maxZ = xyz[2] < grid.getSize()[2]-1 ? xyz[2]+1 : xyz[2];

        for (uint32_t x = minX; x <= maxX; ++x) {
            for (uint32_t y = minY; y <= maxY; ++y) {
                for (uint32_t z = minZ; z <= maxZ; ++z) {
                    size_t j = grid.idx3To1({ x, y, z });
                    if (maxVal[i] < maxVal[j] || (maxVal[i] == maxVal[j] && j < i)) {
                        maxPos[i].x = std::numeric_limits<float>::quiet_NaN();
                        return;
                    }
                }
            }
        }
    });

    const size_t total = grid.getTotalSize();
    for (size_t i = 0; i < total; ++i) {
        if (!std::isnan(maxPos[i].x)) {
            atomPosition.push_back(maxPos[i]);
            atomNormal.push_back(grid.getData(i)->normal);
        }
    }
}

void TriPhasor::saveAtomsToPLY(const char *path, float layerHeight) const {  
    std::ofstream plyFile(path);
    plyFile << "ply\nformat ascii 1.0\nelement vertex " << 2*atomPosition.size() << "\n"
            << "property float x\nproperty float y\nproperty float z\n"
            << "element edge " << atomPosition.size() << "\nproperty int vertex1\nproperty int vertex2\n"
            << "end_header\n";
    
    float l = 0.5f*layerHeight;
    for (size_t i = 0; i < atomPosition.size() ; ++i) {
        plyFile << atomPosition[i].x << " " << atomPosition[i].y << " " << atomPosition[i].z << std::endl;
        plyFile << atomPosition[i].x + l*atomNormal[i].x << " " << atomPosition[i].y + l*atomNormal[i].y << " " << atomPosition[i].z + l*atomNormal[i].z << std::endl;
    }

    for (size_t i = 0; i < atomPosition.size(); ++i) {
        plyFile << 2*i << " " << 2*i+1 << std::endl;
    }
}

void TriPhasor::restrictLayers(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight) {
    float x = 0, y = 0;
    bool frozen = false;
    bool outside = true;
    Vec3 norm(0);

    for (size_t n : gridCoarse->coarseToDense(i)) {
        if (isOutside(gridDense->getData(n)->flags)) continue;
        norm += gridDense->getData(n)->normal;
    }
    gridCoarse->getData(i)->normal = norm.normalize();
        
    for (size_t n : gridCoarse->coarseToDense(i)) {
        if (isOutside(gridDense->getData(n)->flags)) continue;
        outside = false;
        if (isFrozen(gridDense->getData(n)->flags)) {
            frozen = true;
            Vec3 d_i = gridCoarse->getData(i)->normal;
            Vec3 d_j = gridDense->getData(n)->normal;
            float f_i = 1.0f/layerHeight;
            float f_j = 1.0f/layerHeight;
            alignPhases(gridCoarse->getPosition(i), gridDense->getPosition(n), d_i, d_j, f_i, f_j, gridDense->getData(n)->phase[2], x, y);
        }
    }

    gridCoarse->getData(i)->phase[2] = atan2f(y, x);
    setFrozen(gridCoarse->getData(i)->flags, frozen);
    setOutside(gridCoarse->getData(i)->flags, outside);
}

void TriPhasor::prolongLayers(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight) {
    if (isOutside(gridDense->getData(i)->flags) || isFrozen(gridDense->getData(i)->flags)) return;
    size_t n = gridDense->denseToCoarse(i);
    Vec3 d_i = gridDense->getData(i)->normal;
    Vec3 d_j = gridCoarse->getData(n)->normal;
    float f_i = 1.0f/layerHeight;
    float f_j = 1.0f/layerHeight;
    float x = 0, y = 0;
    alignPhases(gridDense->getPosition(i), gridCoarse->getPosition(n), d_i, d_j, f_i, f_j, gridCoarse->getData(n)->phase[2], x, y);
    gridDense->getData(i)->phase[2] = atan2f(y, x);
}

void TriPhasor::smoothLayers(size_t i, GridHierarchy<Data> *grid, float layerHeight) {
    if (isOutside(grid->getData(i)->flags) || isFrozen(grid->getData(i)->flags)) return;
    float x = 0, y = 0;
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags)) continue;
        Vec3 d_i = grid->getData(i)->normal;
        Vec3 d_j = grid->getData(n)->normal;
        float f_i = 1.0f/layerHeight;
        float f_j = 1.0f/layerHeight;
        alignPhases(grid->getPosition(i), grid->getPosition(n), d_i, d_j, f_i, f_j, grid->getData(n)->phase[2], x, y);
    }
    grid->getData(i)->phase[2] = atan2f(y, x);
}

Vec3 TriPhasor::gradSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid) {
    auto xyz = grid.idx1To3(i);
    std::array<uint32_t, 3> xm1yz = xyz[0] > 0 ? std::array<uint32_t, 3>{xyz[0]-1, xyz[1], xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xp1yz = xyz[0] < grid.getSize()[0]-1 ? std::array<uint32_t, 3>{xyz[0]+1, xyz[1], xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xym1z = xyz[1] > 0 ? std::array<uint32_t, 3>{xyz[0], xyz[1]-1, xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyp1z = xyz[1] < grid.getSize()[1]-1 ? std::array<uint32_t, 3>{xyz[0], xyz[1]+1, xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyzm1 = xyz[2] > 0 ? std::array<uint32_t, 3>{xyz[0], xyz[1], xyz[2]-1} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyzp1 = xyz[2] < grid.getSize()[2]-1 ? std::array<uint32_t, 3>{xyz[0], xyz[1], xyz[2]+1} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    float dx = xm1yz[0] == uint32_t(-1) || xp1yz[0] == uint32_t(-1) ? grid.getCellSize() : 2.0f*grid.getCellSize();
    float dy = xym1z[0] == uint32_t(-1) || xyp1z[0] == uint32_t(-1) ? grid.getCellSize() : 2.0f*grid.getCellSize();
    float dz = xyzm1[0] == uint32_t(-1) || xyzp1[0] == uint32_t(-1) ? grid.getCellSize() : 2.0f*grid.getCellSize();
    if (xm1yz[0] == uint32_t(-1)) xm1yz = xyz;
    if (xp1yz[0] == uint32_t(-1)) xp1yz = xyz;
    if (xym1z[0] == uint32_t(-1)) xym1z = xyz;
    if (xyp1z[0] == uint32_t(-1)) xyp1z = xyz;
    if (xyzm1[0] == uint32_t(-1)) xyzm1 = xyz;
    if (xyzp1[0] == uint32_t(-1)) xyzp1 = xyz;
    
    float dfdx = (sdf[grid.idx3To1(xp1yz)]-sdf[grid.idx3To1(xm1yz)])/dx;
    float dfdy = (sdf[grid.idx3To1(xyp1z)]-sdf[grid.idx3To1(xym1z)])/dy;
    float dfdz = (sdf[grid.idx3To1(xyzp1)]-sdf[grid.idx3To1(xyzm1)])/dz;

    return Vec3(dfdx, dfdy, dfdz).normalize();
}


void TriPhasor::initLayers(size_t i, const std::vector<float> &sdf, const std::vector<Vec3> &normals, GridHierarchy<Data> &grid, float layerHeight, float maxTopAngle, float maxBottomAngle, NormalField::Objective objective) {
    Vec3 sdfGrad = gradSDF(i, sdf, grid);

    float cellDiag = cellSizeFromLayerHeight(layerHeight)*sqrt(3);
    bool isBoundary = sdf[i] < 0.5f*cellDiag && sdf[i] > -0.5f*cellDiag;
    bool isDown = sdfGrad.z < 0;
    float theta = acos(isDown ? -sdfGrad.z : sdfGrad.z)/M_PI*180.0f;
    Region region = Region::Other;
    if (isDown && isBoundary && theta < maxBottomAngle) {
        region = Region::Bottom;
    } else if (!isDown && isBoundary && theta < maxTopAngle) {
        region = Region::Top;
    } else if (isBoundary) {
        region = Region::Wall;
    }

    grid.getData(i)->normal = normals[i];
    setRegion(grid.getData(i)->flags, region);
    setOutside(grid.getData(i)->flags, sdf[i] >= 0.5f*cellDiag);
    if (objective == NormalField::Objective::SupportFree) {
        float angle = acosf(fabsf(grid.getData(i)->normal.dot(gradSDF(i, sdf, grid))));
        setFrozen(grid.getData(i)->flags, angle < M_PI/8 && isBoundary && curvatureSDF(i, sdf, grid) < 0.1);
        grid.getData(i)->phase[2] = 2.0f*M_PI*(sdf[i]/layerHeight+0.5f) * (grid.getData(i)->normal.dot(gradSDF(i, sdf, grid)) > 0 ? 1 : -1);
    } else {
        setFrozen(grid.getData(i)->flags, (region == Region::Top || region == Region::Bottom || (grid.getPosition(i).z < cellDiag && theta < 0.1)) && curvatureSDF(i, sdf, grid) < 0.1);
        grid.getData(i)->phase[2] = 2.0f*M_PI*(sdf[i]/layerHeight+0.5f) * (region == Region::Top ? 1 : -1);
    }
}

void TriPhasor::initTangents(size_t i, const std::vector<float> &sdf, const std::vector<Vec3> &normals, GridHierarchy<Data> &grid, float layerHeight, float maxTopAngle, float maxBottomAngle, NormalField::Objective objective) {
    Vec3 sdfGrad = gradSDF(i, sdf, grid);

    float cellDiag = cellSizeFromLayerHeight(layerHeight)*sqrt(3);
    bool isBoundary = sdf[i] < 0.5f*cellDiag && sdf[i] > -0.5f*cellDiag;
    bool isDown = sdfGrad.z < 0;
    float theta = acos(isDown ? -sdfGrad.z : sdfGrad.z)/M_PI*180.0f;
    Region region = Region::Other;
    if (isDown && isBoundary && theta < maxBottomAngle) {
        region = Region::Bottom;
    } else if (!isDown && isBoundary && theta < maxTopAngle) {
        region = Region::Top;
    } else if (isBoundary) {
        region = Region::Wall;
    }

    grid.getData(i)->tangent = normals[i].cross(sdfGrad).normalize();
    setRegion(grid.getData(i)->flags, region);
    setOutside(grid.getData(i)->flags, sdf[i] >= 0.5f*cellDiag);
    setFrozen(grid.getData(i)->flags, getRegion(grid.getData(i)->flags) == Region::Wall && curvatureSDF(i, sdf, grid) < 0.1);
    if (objective == NormalField::Objective::SupportFree) {
        float angle = acosf(fabsf(grid.getData(i)->normal.dot(gradSDF(i, sdf, grid))));
        setFrozen(grid.getData(i)->flags, angle > M_PI/8 && isBoundary && curvatureSDF(i, sdf, grid) < 0.1);
    } else {
        setFrozen(grid.getData(i)->flags, getRegion(grid.getData(i)->flags) == Region::Wall && curvatureSDF(i, sdf, grid) < 0.1);
    }
}

void TriPhasor::restrictTangents(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) {
    Vec3 tang(0);
    bool frozen = false;
    for (size_t n : gridCoarse->coarseToDense(i)) {
        if (isOutside(gridDense->getData(n)->flags)) continue;
        if (isFrozen(gridDense->getData(n)->flags)) {
            frozen = true;
            tang += gridDense->getData(n)->tangent * (tang.dot(gridDense->getData(n)->tangent) < 0 ? -1 : 1);
        }
    }
    gridCoarse->getData(i)->tangent = tang.length2() ? tang.orthonormalize(gridCoarse->getData(i)->normal) : Vec3(0);
    setFrozen(gridCoarse->getData(i)->flags, frozen);
}

void TriPhasor::prolongTangents(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) {
    if (isOutside(gridDense->getData(i)->flags) || isFrozen(gridDense->getData(i)->flags)) return;
    size_t n = gridDense->denseToCoarse(i);
    gridDense->getData(i)->tangent = gridCoarse->getData(n)->tangent.orthonormalize(gridDense->getData(i)->normal);
}

void TriPhasor::smoothTangents(size_t i, GridHierarchy<Data> *grid) {
    if (isOutside(grid->getData(i)->flags) || isFrozen(grid->getData(i)->flags)) return;
    Vec3 tang(0);
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags)) continue;
        tang += grid->getData(n)->tangent * (tang.dot(grid->getData(n)->tangent) < 0 ? -1 : 1);
    }
    grid->getData(i)->tangent = tang.orthonormalize(grid->getData(i)->normal);
}

void TriPhasor::smoothTangentsAll(size_t i, GridHierarchy<Data> *grid) {
    Vec3 tang(0);
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags)) continue;
        tang += grid->getData(n)->tangent * (tang.dot(grid->getData(n)->tangent) < 0 ? -1 : 1);
    }
    grid->getData(i)->tangent = tang.orthonormalize(grid->getData(i)->normal);
}

void TriPhasor::restrictTriphasor(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight) {
    float x0 = 0, y0 = 0;
    float x1 = 0, y1 = 0;
    Vec3 tang(0);
    for (size_t n : gridCoarse->coarseToDense(i)) {
        if (isOutside(gridDense->getData(n)->flags)) continue;
        tang += gridDense->getData(n)->tangent * (tang.dot(gridDense->getData(n)->tangent) < 0 ? -1 : 1);
    }
    gridCoarse->getData(i)->tangent = tang.orthonormalize(gridCoarse->getData(i)->normal);
        
    for (size_t n : gridCoarse->coarseToDense(i)) {
        if (isOutside(gridDense->getData(n)->flags)) continue;
        if (isFrozen(gridDense->getData(n)->flags)) {
            Vec3 d_i = gridCoarse->getData(i)->tangent;
            Vec3 d_j = gridDense->getData(n)->tangent;
            Vec3 d_i_b = gridCoarse->getData(i)->normal.cross(d_i).normalize();
            Vec3 d_j_b = gridDense->getData(n)->normal.cross(d_j).normalize();
            float f_i = 1.0f/layerHeight;
            float f_j = 1.0f/layerHeight;
            alignPhases(gridCoarse->getPosition(i), gridDense->getPosition(n), d_i, d_j, f_i, f_j, gridDense->getData(n)->phase[0], x0, y0);
            alignPhases(gridCoarse->getPosition(i), gridDense->getPosition(n), d_i_b, d_j, f_i, f_j, gridDense->getData(n)->phase[0], x1, y1);
            alignPhases(gridCoarse->getPosition(i), gridDense->getPosition(n), d_i, d_j_b, f_i, f_j, gridDense->getData(n)->phase[1], x0, y0);
            alignPhases(gridCoarse->getPosition(i), gridDense->getPosition(n), d_i_b, d_j_b, f_i, f_j, gridDense->getData(n)->phase[1], x1, y1);
        }
    }

    gridCoarse->getData(i)->phase[0] = atan2f(y0, x0);
    gridCoarse->getData(i)->phase[1] = atan2f(y1, x1);
}

void TriPhasor::prolongTriphasor(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse, float layerHeight) {
    if (isOutside(gridDense->getData(i)->flags)) return;
    size_t n = gridDense->denseToCoarse(i);
    Vec3 d_i = gridDense->getData(i)->tangent;
    Vec3 d_j = gridCoarse->getData(n)->tangent;
    Vec3 d_i_b = gridDense->getData(i)->normal.cross(d_i).normalize();
    Vec3 d_j_b = gridCoarse->getData(n)->normal.cross(d_j).normalize();
    float f_i = 1.0f/layerHeight;
    float f_j = 1.0f/layerHeight;
    float x0 = 0, y0 = 0;
    float x1 = 0, y1 = 0;
    alignPhases(gridDense->getPosition(i), gridCoarse->getPosition(n), d_i, d_j, f_i, f_j, gridCoarse->getData(n)->phase[0], x0, y0);
    alignPhases(gridDense->getPosition(i), gridCoarse->getPosition(n), d_i_b, d_j, f_i, f_j, gridCoarse->getData(n)->phase[0], x1, y1);
    alignPhases(gridDense->getPosition(i), gridCoarse->getPosition(n), d_i, d_j_b, f_i, f_j, gridCoarse->getData(n)->phase[1], x0, y0);
    alignPhases(gridDense->getPosition(i), gridCoarse->getPosition(n), d_i_b, d_j_b, f_i, f_j, gridCoarse->getData(n)->phase[1], x1, y1);

    gridDense->getData(i)->phase[0] = atan2f(y0, x0);
    if (!isFrozen(gridDense->getData(i)->flags)) {
        gridDense->getData(i)->phase[1] = atan2f(y1, x1);
    }
}

void TriPhasor::smoothTriphasor(size_t i, GridHierarchy<Data> *grid, float layerHeight) {
    if (isOutside(grid->getData(i)->flags)) return;
    float x0 = 0, y0 = 0;
    float x1 = 0, y1 = 0;
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags)) continue;
        if (fabsf(grid->getData(i)->phase[2] - grid->getData(n)->phase[2]) > M_PI) continue;
        Vec3 d_i = grid->getData(i)->tangent;
        Vec3 d_j = grid->getData(n)->tangent;
        Vec3 d_i_b = grid->getData(i)->normal.cross(d_i).normalize();
        Vec3 d_j_b = grid->getData(n)->normal.cross(d_j).normalize();
        float f_i = 1.0f/layerHeight;
        float f_j = 1.0f/layerHeight;
        alignPhases(grid->getPosition(i), grid->getPosition(n), d_i, d_j, f_i, f_j, grid->getData(n)->phase[0], x0, y0);
        alignPhases(grid->getPosition(i), grid->getPosition(n), d_i_b, d_j, f_i, f_j, grid->getData(n)->phase[0], x1, y1);
        alignPhases(grid->getPosition(i), grid->getPosition(n), d_i, d_j_b, f_i, f_j, grid->getData(n)->phase[1], x0, y0);
        alignPhases(grid->getPosition(i), grid->getPosition(n), d_i_b, d_j_b, f_i, f_j, grid->getData(n)->phase[1], x1, y1);
    }
    grid->getData(i)->phase[0] = atan2f(y0, x0);
    if (!isFrozen(grid->getData(i)->flags)) {
        grid->getData(i)->phase[1] = atan2f(y1, x1);
    }
}

float TriPhasor::curvatureSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid) {
    auto xyz = grid.idx1To3(i);
    std::array<uint32_t, 3> xm1yz = xyz[0] > 0 ? std::array<uint32_t, 3>{xyz[0]-1, xyz[1], xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xp1yz = xyz[0] < grid.getSize()[0]-1 ? std::array<uint32_t, 3>{xyz[0]+1, xyz[1], xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xym1z = xyz[1] > 0 ? std::array<uint32_t, 3>{xyz[0], xyz[1]-1, xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyp1z = xyz[1] < grid.getSize()[1]-1 ? std::array<uint32_t, 3>{xyz[0], xyz[1]+1, xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyzm1 = xyz[2] > 0 ? std::array<uint32_t, 3>{xyz[0], xyz[1], xyz[2]-1} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyzp1 = xyz[2] < grid.getSize()[2]-1 ? std::array<uint32_t, 3>{xyz[0], xyz[1], xyz[2]+1} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    if (xm1yz[0] == uint32_t(-1)) xm1yz = xyz;
    if (xp1yz[0] == uint32_t(-1)) xp1yz = xyz;
    if (xym1z[0] == uint32_t(-1)) xym1z = xyz;
    if (xyp1z[0] == uint32_t(-1)) xyp1z = xyz;
    if (xyzm1[0] == uint32_t(-1)) xyzm1 = xyz;
    if (xyzp1[0] == uint32_t(-1)) xyzp1 = xyz;

    float x = (1.0f-gradSDF(grid.idx3To1(xm1yz), sdf, grid).dot(gradSDF(grid.idx3To1(xp1yz), sdf, grid)))*0.5f;
    float y = (1.0f-gradSDF(grid.idx3To1(xym1z), sdf, grid).dot(gradSDF(grid.idx3To1(xyp1z), sdf, grid)))*0.5f;
    float z = (1.0f-gradSDF(grid.idx3To1(xyzm1), sdf, grid).dot(gradSDF(grid.idx3To1(xyzp1), sdf, grid)))*0.5f;

    return (x+y+z)/3.0f;
}

void TriPhasor::saveSliceToPLY(const char *path, const GridHierarchy<Data> &grid, float layerHeight) const {
    std::vector<Vec3> pos;
    std::vector<std::array<uint32_t, 4>> faces;
    std::vector<std::array<uint8_t, 3>> colors;

    auto size = grid.getSize();
    size_t z = 0.2f*size[2];
    for (size_t x = 0; x < size[0]-1; ++x) {
        for (size_t y = 0; y < size[1]-1; ++y) {
            if (isOutside(grid.getData(grid.idx3To1(x, y, z))->flags)) continue;
            if (isOutside(grid.getData(grid.idx3To1(x, y, z+1))->flags)) continue;
            if (isOutside(grid.getData(grid.idx3To1(x+1, y, z+1))->flags)) continue;
            if (isOutside(grid.getData(grid.idx3To1(x+1, y, z))->flags)) continue;

            uint32_t curr = pos.size();
            Vec3 p00 = grid.getPosition(grid.idx3To1(x, y, z));
            Vec3 p02 = grid.getPosition(grid.idx3To1(x, y+1, z));
            Vec3 p22 = grid.getPosition(grid.idx3To1(x+1, y+1, z));
            Vec3 p20 = grid.getPosition(grid.idx3To1(x+1, y, z));
            Vec3 p01 = 0.5f*(p00+p02);
            Vec3 p10 = 0.5f*(p00+p20);
            Vec3 p11 = 0.5f*(p00+p22);
            Vec3 p12 = Vec3(0.5f*(p00.x+p22.x), p22.y, 0.5f*(p00.z+p22.z));
            Vec3 p21 = Vec3(p22.x, 0.5f*(p00.y+p22.y), 0.5f*(p00.z+p22.z));
            pos.push_back(p00);
            pos.push_back(p02);
            pos.push_back(p22);
            pos.push_back(p20);
            pos.push_back(p01);
            pos.push_back(p10);
            pos.push_back(p11);
            pos.push_back(p12);
            pos.push_back(p21);

            faces.push_back({curr+0, curr+4, curr+6, curr+5});
            faces.push_back({curr+2, curr+7, curr+6, curr+8});
            faces.push_back({curr+6, curr+7, curr+1, curr+4});
            faces.push_back({curr+6, curr+5, curr+3, curr+8});

            auto color = [&](const Vec3 &position) {
                size_t idx = grid.getIndex(position);
                Vec3 dirT = grid.getData(idx)->tangent;
                Vec3 dirN = grid.getData(idx)->normal;
                Vec3 dirB = dirN.cross(dirT).normalize();
                float phaseT = grid.getData(idx)->phase[0];
                float phaseB = grid.getData(idx)->phase[1];
                float phaseN = grid.getData(idx)->phase[2];
                float valT = cosf(2.0*M_PI/layerHeight*dirT.dot(position-grid.getPosition(idx))+phaseT);
                float valB = cosf(2.0*M_PI/layerHeight*dirB.dot(position-grid.getPosition(idx))+phaseB);
                float valN = cosf(2.0*M_PI/layerHeight*dirN.dot(position-grid.getPosition(idx))+phaseN);

                return viridis(0.5f*(valT+valB+valN)/3+0.5f);
            };

            for (uint32_t i = 0; i < 9; ++i) {
                colors.push_back(color(pos[curr+i]));
            }
        }
    }

    uint32_t usedVerts = 0;
    std::unordered_map<Vec3, uint32_t> vertsID;
    std::unordered_set<uint32_t> usedVert;
    for (size_t i = 0; i < pos.size(); ++i) {
        if (!vertsID.count(pos[i])) {
            vertsID[pos[i]] = usedVerts++;
            usedVert.insert(i);
        }
    }

    for (size_t i = 0; i < faces.size(); ++i) {
        faces[i][0] = vertsID[pos[faces[i][0]]];
        faces[i][1] = vertsID[pos[faces[i][1]]];
        faces[i][2] = vertsID[pos[faces[i][2]]];
        faces[i][3] = vertsID[pos[faces[i][3]]];
    }

    std::ofstream plyFile(path);
    plyFile << "ply\nformat binary_little_endian 1.0\nelement vertex " << usedVerts << "\n"
            << "property float x\nproperty float y\nproperty float z\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            << "element face " << faces.size() << "\nproperty list uchar uint vertex_indices\n"
            << "end_header\n";
    plyFile.close();
    plyFile = std::ofstream(path, std::ios::out | std::ios::binary | std::ios::app);

    for (size_t i = 0; i < pos.size(); ++i) {
        if (!usedVert.count(i)) continue;
        plyFile.write(reinterpret_cast<const char*>(&pos[i].x), 4);
        plyFile.write(reinterpret_cast<const char*>(&pos[i].y), 4);
        plyFile.write(reinterpret_cast<const char*>(&pos[i].z), 4);
        plyFile.write(reinterpret_cast<const char*>(&colors[i][0]), 1);
        plyFile.write(reinterpret_cast<const char*>(&colors[i][1]), 1);
        plyFile.write(reinterpret_cast<const char*>(&colors[i][2]), 1);
    }

    for (const auto &face : faces) {
        uint8_t four = 4;
        plyFile.write(reinterpret_cast<const char*>(&four), 1);
        plyFile.write(reinterpret_cast<const char*>(&face[0]), 4);
        plyFile.write(reinterpret_cast<const char*>(&face[1]), 4);
        plyFile.write(reinterpret_cast<const char*>(&face[2]), 4);
        plyFile.write(reinterpret_cast<const char*>(&face[3]), 4);
    }
}
