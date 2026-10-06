// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "toolpath.h"
#include "curve.h"
#include "bvhpoint.h"
#include "bvhtriangle.h"
#include <iomanip>
#include <string>
#include <sstream>

Toolpath::Toolpath(const std::vector<Mesh> &layers, const SDF &sdf, const std::vector<Vec3> &points, const std::vector<Vec3> &normals, float width, float height, uint32_t hardCodedField, uint32_t zigzagOffset, bool holeClosing, bool quiet) {
    size_t travels = 0;
    BVHPoint bvh(points);
    Vec3 lastPoint(0);
    float higherZ = -std::numeric_limits<float>::infinity();
    for (auto &point : points) {
        if (point.z > higherZ) {
            higherZ = point.z;
            lastPoint = Vec3(point.x, point.y, 0);
        }
    }

    float maxZ = 0.0f;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (!quiet) printProgress(float(i)/(layers.size()-1));
        if (!layers[i].vertices.size() || !layers[i].triangles.size()) continue;

        std::vector<float> directions(layers[i].vertices.size()/3, ((i+zigzagOffset) % 4) * 0.25*M_PI);

        // User-defined tool-tangents for the Airfoil (constant per-layer).
        if (hardCodedField == 2 /* Airfoil */) {
            if (i == 0 || i == 6 || i == 301 || i == 311) {
                directions = std::vector<float>(layers[i].vertices.size()/3, M_PI * 0.5);
            }
        }

        // User-defined tool-tangents for the House (loaded from per-layer texture).
        if (hardCodedField == 3 /* House */) {
            switch (i) {
            case 0:
                directionsFromTexture("data/images/cubic.pgm", directions, layers[i].vertices, sdf.getModelSize());
                break;
            case 3:
                directionsFromTexture("data/images/siggraph.pgm", directions, layers[i].vertices, sdf.getModelSize());
                break;
            case 188:
            case 227:
            case 228:
                directionsFromTexture("data/images/right.pgm", directions, layers[i].vertices, sdf.getModelSize());
                break;
            case 229:
                directionsFromTexture("data/images/front.pgm", directions, layers[i].vertices, sdf.getModelSize());
                break;
            case 230:
                directionsFromTexture("data/images/tiles.pgm", directions, layers[i].vertices, sdf.getModelSize());
                break;
            default:
                break;
            }
        }

        Curve curve(layers[i].vertices.data(), layers[i].vertices.size()/3, layers[i].triangles.data(), layers[i].triangles.size()/3,
                    width, sdf, directions.data(), Stripe::Printing, true, holeClosing, true);
        auto cycles = curve.getCycles();

        for (const auto &cycle : cycles) {
            if (cycle.size() < 2) continue;
            if (toolpath.size()) lastPoint = toolpath.back().position;
            size_t closestIdx = 0;
            float closestDist2 = std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < cycle.size(); ++j) {
                float dist2 = (cycle[j][0]-lastPoint).length2();
                if (dist2 < closestDist2) {
                    closestDist2 = dist2;
                    closestIdx = j;
                }
            }

            if (toolpath.size() && sqrt(closestDist2) > 2.0f*width) {
                Vec3 pos0 =  toolpath.back().position;
                Vec3 norm0 = toolpath.back().normal;
                Vec3 pos1 = cycle[closestIdx][0];
                Vec3 norm1 = normals[bvh.nearestPointIndex(pos1)];
                float z = std::max(maxZ, pos1.z) + 4.0*width;
                pos0 += ((z-pos0.z)/fabsf(norm0.z))*norm0;
                pos1 += ((z-pos1.z)/fabsf(norm1.z))*norm1;

                if (norm0.z < cosf(0.25f*M_PI) || norm1.z < cosf(0.25f*M_PI)) {
                    Vec3 minAABB, maxAABB;
                    bvh.getAABB(minAABB, maxAABB);
                    pos0 = pos0.clamp(minAABB-4.0*width, maxAABB+4.0*width);
                    pos1 = pos1.clamp(minAABB-4.0*width, maxAABB+4.0*width);
                }

                toolpath.push_back({pos0+0.5*height*norm0, norm0, false});
                toolpath.push_back({pos0+0.5*height*norm0, Vec3(0,0,1), false});
                toolpath.push_back({pos1+0.5*height*norm1, Vec3(0,0,1), false});
                toolpath.push_back({pos1+0.5*height*norm1, norm1, false});
                travels++;
            }

            for (size_t j = 0; j < cycle.size(); ++j) {
                size_t idx = (closestIdx + j) % cycle.size();
                Vec3 pos = cycle[idx][0];
                Vec3 norm = normals[bvh.nearestPointIndex(cycle[idx][0])];
                maxZ = std::max(maxZ, pos.z);
                toolpath.push_back({pos+0.5*height*norm, norm, j > 0});
            }
        }
    }
    if (toolpath.size()) {
        Vec3 pos(toolpath.back().position.x, toolpath.back().position.y, maxZ+4.0f*height);
        toolpath.push_back({ pos, Vec3(0,0,1), false });
    }

    if (!quiet) {
        float depositionLength = 0.0f, totalLength = 0.0f;
        for (size_t i = 1; i < toolpath.size(); ++i) {
            if (toolpath[i].deposition)
                depositionLength += (toolpath[i].position-toolpath[i-1].position).length();
            totalLength += (toolpath[i].position-toolpath[i-1].position).length();
        }
        std::cout << std::fixed << std::setprecision(0) << "Toolpath length: " << totalLength << " mm (travels: " << std::setprecision(1)
        << (totalLength-depositionLength)/totalLength*100.0f << std::setprecision(3) << "%)" << std::endl;
        std::cout << "Long travels: " << travels << std::endl;
    }
}

void Toolpath::smoothNormals(size_t iterations) {
    std::vector<WayPoint> toolpathBack(toolpath);
    for (size_t iter = 0; iter < iterations; ++iter) {
        Parallel::For(0, toolpath.size(), [iter, this, &toolpathBack](size_t i) {
            auto &front = (iter % 2 ? toolpathBack : toolpath);
            const auto &back = (iter % 2 == 0 ? toolpathBack : toolpath);
            Vec3 norm(back[i].normal);
            if (i > 0 && toolpath[i-1].deposition) norm += back[i-1].normal;
            if (i < toolpath.size()-1 && toolpath[i+1].deposition) norm += back[i+1].normal;
            front[i].normal = norm.length() > std::numeric_limits<float>::epsilon() ? norm.normalize() : back[i].normal;
        });
    }
    toolpathBack.clear();
}

void Toolpath::smoothPositions(size_t iterations, const SDF &sdf, float width, bool holeClosing) {
    std::vector<WayPoint> toolpathBack(toolpath);
    for (size_t iter = 0; iter < iterations; ++iter) {
        Parallel::For(0, toolpath.size(), [iter, this, &toolpathBack, &sdf, width, holeClosing](size_t i) {
            auto &front = (iter % 2 ? toolpathBack : toolpath);
            const auto &back = (iter % 2 == 0 ? toolpathBack : toolpath);
            if (i > 0 && i < toolpath.size()-1 && toolpath[i+1].deposition && toolpath[i-1].deposition && toolpath[i].deposition) {
                if (!holeClosing || sdf.getVal(back[i].position) < -width || fabsf(sdf.getGrad(back[i].position).dot(back[i].normal)) > 0.7f) {
                    front[i].position = (back[i-1].position+back[i].position+back[i+1].position)/3;
                }
            }
        });
    }
    toolpathBack.clear();
}

void Toolpath::singularityCone(float angle) {
    Vec3 lastNorm(0,0,1);
    for (auto &waypoint : toolpath) {
        if (acos(waypoint.normal.dot(lastNorm))/M_PI*180.0 > angle) {
            lastNorm = waypoint.normal;
        } else {
            waypoint.normal = lastNorm;
        }
    }
}

void Toolpath::directionsFromTexture(const char *path, std::vector<float> &directions, const std::vector<float> &positions, const Vec3 &modelSize) const {
    std::ifstream file(path);
    if (!file.good()) {
        std::cout << "Fatal Error: could not open file " << path << "!" << std::endl;
        exit(1);
    }
    std::string line;
    for (uint32_t j = 0; j < 3; ++j) {
        std::getline(file, line);
    }
    int32_t texelCountX, texelCountY, levelCount, value;
    std::istringstream lineStream(line);
    lineStream >> texelCountX >> texelCountY;
    std::getline(file, line);
    lineStream.clear();
    lineStream.str(line);
    lineStream >> levelCount;

    std::vector<float> directionsTexture;
    while (std::getline(file, line)) {
        lineStream.clear();
        lineStream.str(line);
        while (lineStream >> value) {
            directionsTexture.push_back(float(value)/(levelCount-1)*M_PI);
        }
    }

    for (size_t j = 0; j < directions.size(); ++j) {
        float posX = positions[3*j+0], posY = positions[3*j+1];
        int32_t texelU = std::min(std::max(int32_t(posX/modelSize.x * texelCountX), 0), texelCountX-1);
        int32_t texelV = std::min(std::max(int32_t(posY/modelSize.y * texelCountY), 0), texelCountY-1);
        directions[j] = directionsTexture[texelV*texelCountY + texelU];
    }
}


void Toolpath::tessellateNormals(float angle) {
    std::vector<WayPoint> toolpathTessellated;
    toolpathTessellated.reserve(toolpath.size());
    toolpathTessellated.push_back(toolpath[0]);
    for (size_t i = 1; i < toolpath.size(); ++i) {
        float angleDistance = acos(std::clamp(toolpath[i-1].normal.dot(toolpath[i].normal), -1.0f, 1.0f))/M_PI*180;
        if (angleDistance > angle) {
            uint32_t stepsCount = ceil(angleDistance/angle);
            for (uint32_t j = 1; j < stepsCount+1; ++j) {
                float t = float(j)/stepsCount;

                WayPoint point;
                point.position = toolpath[i-1].position*(1-t) + toolpath[i].position*t;
                point.normal = (toolpath[i-1].normal*(1-t) + toolpath[i].normal*t).normalize();
                point.deposition = toolpath[i].deposition;
                toolpathTessellated.push_back(point);   
            }
        } else {
            toolpathTessellated.push_back(toolpath[i]);
        }
    }
    toolpath = std::move(toolpathTessellated);
}

void Toolpath::saveToCSV(const char *path) const {
    std::ofstream toolpathFile(path);
    toolpathFile << toolpath.size() << std::endl;
    toolpathFile << std::setprecision(6);
    for (const auto &waypoint : toolpath) {
        toolpathFile << (waypoint.deposition ? "D " : "T ")
                     << waypoint.position.x << " " << waypoint.position.y << " " << waypoint.position.z << " "
                     << waypoint.normal.x << " " << waypoint.normal.y << " " << waypoint.normal.z << std::endl;
    }
    toolpathFile.close();
}

void Toolpath::saveToPLY(const char *path, float layerHeight) const {
    std::vector<Vec3> plyVerts;
    std::vector<Vec3> plyNorm;
    std::vector<uint8_t> endPoints;
    std::vector<std::array<uint8_t, 3>> plyColors;
    std::vector<std::array<uint32_t, 2>> plyEdges;

    float totalLength = 0.0f;
    for (size_t i = 1; i < toolpath.size(); ++i) {
        if (!toolpath[i].deposition)
            continue;
        totalLength += (toolpath[i].position-toolpath[i-1].position).length();
    }

    float length = 0.0f;
    bool traveled = false;
    for (size_t i = 0; i < toolpath.size(); ++i) {
        if (!toolpath[i].deposition) {
            traveled = true;
            continue;
        };
        if (traveled) {
            plyVerts.push_back(toolpath[i-1].position);
            plyNorm.push_back(toolpath[i-1].normal);
            plyColors.push_back(turbo(length/totalLength));
            endPoints.push_back(true);
            traveled = false;
        }
        length += (toolpath[i].position-toolpath[i-1].position).length();
        plyVerts.push_back(toolpath[i].position);
        plyNorm.push_back(toolpath[i].normal);
        plyColors.push_back(turbo(length/totalLength));
        endPoints.push_back(i == toolpath.size()-1 || !toolpath[i+1].deposition);

        plyEdges.push_back({uint32_t(plyVerts.size()-2), uint32_t(plyVerts.size()-1)});
    }

    std::ofstream plyFile(path);
    plyFile << "ply\nformat binary_little_endian 1.0\nelement vertex " << plyVerts.size() << "\n"
            << "property float x\nproperty float y\nproperty float z\n"
            << "property float nx\nproperty float ny\nproperty float nz\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            << "property uchar endpoint\n"
            << "element edge " << plyEdges.size() << "\nproperty int vertex1\nproperty int vertex2\n"
            << "end_header\n";
    plyFile.close();
    plyFile = std::ofstream(path, std::ios::out | std::ios::binary | std::ios::app);

    for (size_t i = 0; i < plyVerts.size(); ++i) {
        plyFile.write(reinterpret_cast<const char*>(&plyVerts[i].x), 4);
        plyFile.write(reinterpret_cast<const char*>(&plyVerts[i].y), 4);
        plyFile.write(reinterpret_cast<const char*>(&plyVerts[i].z), 4);
        plyFile.write(reinterpret_cast<const char*>(&plyNorm[i].x), 4);
        plyFile.write(reinterpret_cast<const char*>(&plyNorm[i].y), 4);
        plyFile.write(reinterpret_cast<const char*>(&plyNorm[i].z), 4);
        plyFile.write(reinterpret_cast<const char*>(&plyColors[i][0]), 1);
        plyFile.write(reinterpret_cast<const char*>(&plyColors[i][1]), 1);
        plyFile.write(reinterpret_cast<const char*>(&plyColors[i][2]), 1);
        plyFile.write(reinterpret_cast<const char*>(&endPoints[i]), 1);
    }

    for (const auto &edge : plyEdges) {
        plyFile.write(reinterpret_cast<const char*>(&edge[0]), 4);
        plyFile.write(reinterpret_cast<const char*>(&edge[1]), 4);
    }
}