// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "normal.h"
#include "parallel.h"
#include <cassert>

NormalField::NormalField(const SDF &sdf, float layerHeight, float maxTopAngle, float maxBottomAngle, Objective objective, uint32_t hardCodedField, float numberOfCover) {
    GridHierarchy<Data> grid({sdf.getSize(0), sdf.getSize(1), sdf.getSize(2)}, sdf.getCellSize(), Vec3(-2*sdf.getCellSize()));

    // User-defined tool-orientation field for the Dragon.
    // Note: to be used with the support free objective for the correct phase constraint.
    if (hardCodedField == 1 /* Dragon */) {
        assert(objective == Objective::SupportFree);

        auto pointAbovePlane = [](const Vec3 &pos, const Vec3 &planePos, const Vec3 &normal) {
            return (pos-planePos).dot(normal) > 0;
        };

        normals.resize(grid.getTotalSize());
        Parallel::For(0, normals.size(), [this, &grid, &pointAbovePlane](size_t i) {
            if (pointAbovePlane(grid.getPosition(i), Vec3(22.344, 0.0, 0.0), Vec3(sin(-70.0/180.0*M_PI), 0.0, cos(-70.0/180.0*M_PI))) &&
                pointAbovePlane(grid.getPosition(i), Vec3(0.0, 0.0, 22.144), Vec3(sin(-10.883/180.0*M_PI), 0.0, cos(-10.883/180.0*M_PI)))) {
                float theta = 30.0 / 180.0 * M_PI;
                if (pointAbovePlane(grid.getPosition(i), Vec3(12.0, 0.0, 52.75), Vec3(sin(53.31/180.0*M_PI), 0.0, cos(53.31/180.0*M_PI)))) {
                    float phi = 0.0;
                    normals[i] = Vec3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
                } else {
                    float phi = M_PI;
                    normals[i] = Vec3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
                }
            } else {
                normals[i] = Vec3(0.0, 0.0, 1.0);
            }     
        });

        return;
    }

    if (objective == Objective::ConformalTop || objective == Objective::ConformalSmooth || objective == Objective::ConformalNearest) {
        Parallel::For(0, grid.getTotalSize(), [this, &sdf, &grid, layerHeight, maxTopAngle, maxBottomAngle](size_t i) {
            init(i, sdf.getField(), grid, layerHeight, maxTopAngle, maxBottomAngle);
        });

        std::vector<uint8_t> frozenBack(grid.getTotalSize(), 1);
        Parallel::For(0, grid.getTotalSize(), [this, &grid, &frozenBack](size_t i) {
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
                        if (x == xyz[0] && y == xyz[1] && z == xyz[2]) continue;
                        size_t j = grid.idx3To1({x, y, z});
                        if (!isFrozen(grid.getData(j)->flags)) {
                            frozenBack[i] = false;
                            return;
                        }
                    }
                }
            }
        });

        Parallel::For(0, grid.getTotalSize(), [this, &grid, &frozenBack](size_t i) {
            auto xyz = grid.idx1To3(i);
            uint32_t minX = xyz[0] > 0 ? xyz[0]-1 : xyz[0];
            uint32_t maxX = xyz[0] < grid.getSize()[0]-1 ? xyz[0]+1 : xyz[0];
            uint32_t minY = xyz[1] > 0 ? xyz[1]-1 : xyz[1];
            uint32_t maxY = xyz[1] < grid.getSize()[1]-1 ? xyz[1]+1 : xyz[1];
            uint32_t minZ = xyz[2] > 0 ? xyz[2]-1 : xyz[2];
            uint32_t maxZ = xyz[2] < grid.getSize()[2]-1 ? xyz[2]+1 : xyz[2];
            setFrozen(grid.getData(i)->flags, false);

            for (uint32_t x = minX; x <= maxX; ++x) {
                for (uint32_t y = minY; y <= maxY; ++y) {
                    for (uint32_t z = minZ; z <= maxZ; ++z) {
                        if (x == xyz[0] && y == xyz[1] && z == xyz[2]) continue;
                        size_t j = grid.idx3To1({x, y, z});
                        if (frozenBack[j]) {
                            setFrozen(grid.getData(i)->flags, true);
                            return;
                        }
                    }
                }
            }
        });

        grid.smooth(smoothFrozen, 2);

        if (objective == Objective::ConformalSmooth) {
            grid.optimize(smooth, restrict, prolong);
        } else {
            Parallel::ForProgress(0, grid.getTotalSize(), [this, &sdf, &grid](size_t i) {
                grid.getData(i)->distance = isFrozen(grid.getData(i)->flags) ? fabsf(sdf.getField()[i]) : std::numeric_limits<float>::infinity();
            });

            grid.smoothConvergence([](size_t i, GridHierarchy<Data> *grid) {
                if (isOutside(grid->getData(i)->flags)) return false;
                bool changed = false;
                for (size_t n : grid->getNeighborhood(i)) {
                    if (isOutside(grid->getData(n)->flags)) continue;
                    float distAB = (grid->getPosition(i)-grid->getPosition(n)).length();
                    if (grid->getData(i)->distance > grid->getData(n)->distance + distAB) {
                        grid->getData(i)->distance = grid->getData(n)->distance + distAB;
                        grid->getData(i)->normal = grid->getData(n)->normal;
                        changed = true;
                    }
                }
                return changed;
            });

            if (objective == Objective::ConformalTop) {
                Parallel::For(0, grid.getTotalSize(), [layerHeight, numberOfCover, &grid](size_t i) {
                    if (grid.getData(i)->distance > numberOfCover*layerHeight) {
                        grid.getData(i)->normal = Vec3(0,0,1);
                    }
                });
            }
        }
    } else if (objective == Objective::SupportFree) {
        Parallel::For(0, grid.getTotalSize(), [this, layerHeight, maxTopAngle, maxBottomAngle, &sdf, &grid](size_t i) {
            init(i, sdf.getField(), grid, layerHeight, maxTopAngle, maxBottomAngle);
            grid.getData(i)->normal = Vec3(0,0,1);
            grid.getData(i)->distance = grid.getPosition(i).z < 2.0f*layerHeight ? fabsf(grid.getPosition(i).z) : std::numeric_limits<float>::infinity();
            setFrozen(grid.getData(i)->flags, grid.getPosition(i).z < 2.0f*layerHeight);
        });

        std::vector<float> distanceBack(grid.getTotalSize());
        bool changed = true;
        while (changed) {
            changed &= Parallel::ForAny(0, grid.getTotalSize(), [&grid, &distanceBack](size_t i) {
                distanceBack[i] = grid.getData(i)->distance;
                bool changed = false;
                if (isOutside(grid.getData(i)->flags)) return false;

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
                            size_t j = grid.idx3To1({x, y, z});
                            float distAB = (grid.getPosition(i)-grid.getPosition(j)).length();
                            if (distanceBack[i] > grid.getData(j)->distance + distAB) {
                                distanceBack[i] = grid.getData(j)->distance + distAB;
                                changed = true;
                            }
                        }
                    }
                }
                return changed;
            });

            changed &= Parallel::ForAny(0, grid.getTotalSize(), [&grid, &distanceBack](size_t i) {
                grid.getData(i)->distance = distanceBack[i];
                bool changed = false;
                if (isOutside(grid.getData(i)->flags)) return false;

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
                            size_t j = grid.idx3To1({x, y, z});
                            float distAB = (grid.getPosition(i)-grid.getPosition(j)).length();
                            if (grid.getData(i)->distance > distanceBack[j] + distAB) {
                                grid.getData(i)->distance = distanceBack[j] + distAB;
                                changed = true;
                            }
                        }
                    }
                }
                return changed;
            });
        }

        Parallel::ForProgress(0, grid.getTotalSize(), [this, layerHeight, maxTopAngle, &sdf, &grid](size_t i) {
            if (isOutside(grid.getData(i)->flags) || grid.getPosition(i).z < 2.0f*layerHeight) return;
            Vec3 norm = gradDist(i, grid);
            float theta = acosf(norm.z);
            float phi = atan2(norm.y, norm.x);
            if (theta/M_PI*180.0f > maxTopAngle) theta = maxTopAngle/180.0f*M_PI;
            norm = Vec3(cosf(phi)*sinf(theta), sinf(phi)*sinf(theta), cosf(theta));
            grid.getData(i)->normal = norm;
        });
        
        grid.smooth(smooth, 128);
    }

    normals.resize(grid.getTotalSize());
    Parallel::For(0, normals.size(), [this, &grid, maxTopAngle](size_t i) {
        grid.getData(i)->normal = grid.getData(i)->normal.normalize();
        float theta = acosf(std::max(-1.0f, std::min(1.0f, grid.getData(i)->normal.z)));
        float phi = atan2(grid.getData(i)->normal.y, grid.getData(i)->normal.x);
        if (theta/M_PI*180.0f > maxTopAngle) theta = maxTopAngle/180.0f*M_PI;
        normals[i]  = Vec3(cosf(phi)*sinf(theta), sinf(phi)*sinf(theta), cosf(theta));
    });

    /*grid.saveVec3ToAttributePLY("../../data/test.ply", [layerHeight, &sdf](size_t i, const GridHierarchy<Data> *grid, bool &save) {
        save = grid->idx1To3(i)[0] % 4 == 0 && grid->idx1To3(i)[2] % 4 == 0;
        save &= sdf.getField()[i] < 0 && grid->idx1To3(i)[1] == grid->getSize()[1]/2;
        return grid->getData(i)->normal;
    });*/
}

void NormalField::restrict(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) {
    Vec3 norm(0);
    bool frozen = false;
    bool outside = true;
    for (size_t n : gridCoarse->coarseToDense(i)) {
        if (isOutside(gridDense->getData(n)->flags)) continue;
        outside = false;
        if (isFrozen(gridDense->getData(n)->flags)) {
            frozen = true;
            norm += gridDense->getData(n)->normal;
        }
    }
    gridCoarse->getData(i)->normal = norm.length2() ? norm.normalize() : Vec3(0);
    setFrozen(gridCoarse->getData(i)->flags, frozen);
    setOutside(gridCoarse->getData(i)->flags, outside);
};

void NormalField::prolong(size_t i, GridHierarchy<Data> *gridDense, GridHierarchy<Data> *gridCoarse) {
    if (isOutside(gridDense->getData(i)->flags) || isFrozen(gridDense->getData(i)->flags)) return;
    size_t n = gridDense->denseToCoarse(i);
    gridDense->getData(i)->normal = gridCoarse->getData(n)->normal;
};

void NormalField::smooth(size_t i, GridHierarchy<Data> *grid) {
    if (isOutside(grid->getData(i)->flags) || isFrozen(grid->getData(i)->flags)) return;
    Vec3 norm(0);
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags)) continue;
        norm += grid->getData(n)->normal;
    }
    grid->getData(i)->normal = norm.normalize();
};

void NormalField::smoothAll(size_t i, GridHierarchy<Data> *grid) {
    if (isOutside(grid->getData(i)->flags)) return;
    Vec3 norm(0);
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags)) continue;
        norm += grid->getData(n)->normal;
    }
    grid->getData(i)->normal = norm.normalize();
};

void NormalField::smoothFrozen(size_t i, GridHierarchy<Data> *grid) {
    if (isOutside(grid->getData(i)->flags) || !isFrozen(grid->getData(i)->flags)) return;
    Vec3 norm(0);
    for (size_t n : grid->getNeighborhood(i)) {
        if (isOutside(grid->getData(n)->flags) || !isFrozen(grid->getData(n)->flags)) continue;
        norm += grid->getData(n)->normal;
    }
    if (norm.length2() > 1e-4) {
        grid->getData(i)->normal = norm.normalize();
    }
};

void NormalField::init(size_t i, const std::vector<float> &sdf, GridHierarchy<Data> &grid, float layerHeight, float maxTopAngle, float maxBottomAngle) {
    Vec3 grad = gradSDF(i, sdf, grid);
    bool isBoundary = sdf[i] < layerHeight && sdf[i] > -layerHeight;
    bool isDown = grad.z < 0;
    float theta = acos(isDown ? -grad.z : grad.z)/M_PI*180.0f;
    Region region = Region::Other;
    if (isDown && isBoundary && theta < maxBottomAngle) {
        region = Region::Bottom;
    } else if (!isDown && isBoundary && theta < maxTopAngle) {
        region = Region::Top;
    } else if (isBoundary) {
        region = Region::Wall;
    }

    setRegion(grid.getData(i)->flags, region);
    grid.getData(i)->normal = grid.getPosition(i).z < 2.0*layerHeight ? Vec3(0,0,1) : grad.normalize();
    if (grid.getData(i)->normal.z < 0) grid.getData(i)->normal = -grid.getData(i)->normal;
    setOutside(grid.getData(i)->flags, sdf[i] >= layerHeight);
    setFrozen(grid.getData(i)->flags, ((region == Region::Top || region == Region::Bottom) || grid.getPosition(i).z < 2.0*layerHeight)
        && sdf[i] < layerHeight && curvatureSDF(i, sdf, grid) < 0.1);
}

Vec3 NormalField::gradSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid) {
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

Vec3 NormalField::gradDist(size_t i, const GridHierarchy<Data> &grid) {
    auto xyz = grid.idx1To3(i);
    std::array<uint32_t, 3> xm1yz = xyz[0] > 0 ? std::array<uint32_t, 3>{xyz[0]-1, xyz[1], xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xp1yz = xyz[0] < grid.getSize()[0]-1 ? std::array<uint32_t, 3>{xyz[0]+1, xyz[1], xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xym1z = xyz[1] > 0 ? std::array<uint32_t, 3>{xyz[0], xyz[1]-1, xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyp1z = xyz[1] < grid.getSize()[1]-1 ? std::array<uint32_t, 3>{xyz[0], xyz[1]+1, xyz[2]} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyzm1 = xyz[2] > 0 ? std::array<uint32_t, 3>{xyz[0], xyz[1], xyz[2]-1} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    std::array<uint32_t, 3> xyzp1 = xyz[2] < grid.getSize()[2]-1 ? std::array<uint32_t, 3>{xyz[0], xyz[1], xyz[2]+1} : std::array<uint32_t, 3>{uint32_t(-1), uint32_t(-1), uint32_t(-1)};
    if (xm1yz[0] != uint32_t(-1) && isOutside(grid.getData(grid.idx3To1(xm1yz))->flags)) xm1yz[0] = uint32_t(-1);
    if (xp1yz[0] != uint32_t(-1) && isOutside(grid.getData(grid.idx3To1(xp1yz))->flags)) xp1yz[0] = uint32_t(-1);
    if (xym1z[0] != uint32_t(-1) && isOutside(grid.getData(grid.idx3To1(xym1z))->flags)) xym1z[0] = uint32_t(-1);
    if (xyp1z[0] != uint32_t(-1) && isOutside(grid.getData(grid.idx3To1(xyp1z))->flags)) xyp1z[0] = uint32_t(-1);
    if (xyzm1[0] != uint32_t(-1) && isOutside(grid.getData(grid.idx3To1(xyzm1))->flags)) xyzm1[0] = uint32_t(-1);
    if (xyzp1[0] != uint32_t(-1) && isOutside(grid.getData(grid.idx3To1(xyzp1))->flags)) xyzp1[0] = uint32_t(-1);

    float dx = xm1yz[0] == uint32_t(-1) || xp1yz[0] == uint32_t(-1) ? grid.getCellSize() : 2.0f*grid.getCellSize();
    float dy = xym1z[0] == uint32_t(-1) || xyp1z[0] == uint32_t(-1) ? grid.getCellSize() : 2.0f*grid.getCellSize();
    float dz = xyzm1[0] == uint32_t(-1) || xyzp1[0] == uint32_t(-1) ? grid.getCellSize() : 2.0f*grid.getCellSize();
    if (xm1yz[0] == uint32_t(-1)) xm1yz = xyz;
    if (xp1yz[0] == uint32_t(-1)) xp1yz = xyz;
    if (xym1z[0] == uint32_t(-1)) xym1z = xyz;
    if (xyp1z[0] == uint32_t(-1)) xyp1z = xyz;
    if (xyzm1[0] == uint32_t(-1)) xyzm1 = xyz;
    if (xyzp1[0] == uint32_t(-1)) xyzp1 = xyz;
    
    float dfdx = (grid.getData(grid.idx3To1(xp1yz))->distance-grid.getData(grid.idx3To1(xm1yz))->distance)/dx;
    float dfdy = (grid.getData(grid.idx3To1(xyp1z))->distance-grid.getData(grid.idx3To1(xym1z))->distance)/dy;
    float dfdz = (grid.getData(grid.idx3To1(xyzp1))->distance-grid.getData(grid.idx3To1(xyzm1))->distance)/dz;

    return Vec3(dfdx, dfdy, dfdz).normalize();
}


float NormalField::curvatureSDF(size_t i, const std::vector<float> &sdf, const GridHierarchy<Data> &grid) {
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
