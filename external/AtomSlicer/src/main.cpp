// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include <chrono>
#include <iostream>
#include <string>
#include "slicer.h"
#include "stlloader.h"

int main (int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [mesh.stl] <options>\n"  << std::endl
        << "Options:" << std::endl
        << "[-d nozzle_diameter] : Nozzle diameter in mm  [default: 0.6]" << std::endl
        << "[-h layer_height_factor] : Ratio between layer height and nozzle width [default: 0.5]" << std::endl
        << "[-t top_tilt_angle] : Maximum tilting angle for top surfaces in degrees [default: 30]" << std::endl
        << "[-b bottom_tilt_angle] : Maximum tilting angle for bottom surfaces in degrees [default: 2]" << std::endl
        << "[-3 3axis.gcode] : 3-axis GCode path, the normals are still provided in a comment on each line" << std::endl
        << "[-5 5axis.gcode] : 5-axis GCode path, for the 3Z kinematics RatRig" << std::endl
        << "[-p toolpath.ply] : Toolpath as PLY polyline" << std::endl
        << "[-i infill_type] : Infill type:" << std::endl
        << "    0 -> Full" << std::endl
        << "    1 -> Hollow" << std::endl
        << "    2 -> Grid" << std::endl
        << "    3 -> Cubic" << std::endl
        << "    4 -> Gyroid [default]" << std::endl
        << "[-o tool_orientation] : Objective for the tool orientation:" << std::endl
        << "    0 -> Conformal, smoothly interpolated" << std::endl
        << "    1 -> Conformal on top and bottom surfaces, planar otherwise [default]" << std::endl
        << "    2 -> Support free" << std::endl
        << "    3 -> Oriented as the closest conformal surface" << std::endl
        << "[-u user_defined] : Hard-coded direction field for specific models:" << std::endl
        << "    0 -> None [default]" << std::endl
        << "    1 -> Dragon tool orientations (Fig. 19)" << std::endl
        << "    2 -> Airfoil tangents (Fig. 30)" << std::endl
        << "    3 -> House tangents (Fig. 31)" << std::endl
        << "[-c number_of_cover] : Number of covers for the conformal on top and bottom surfaces objective [default: 4]" << std::endl
        << "[-z zigzag_offset] : Zigzag direction offset [default: 0]" << std::endl
        << "[-C collision_angle] : Cone angle for nozzle collision detection [default: 95]" << std::endl
        << "[-U] : Force partitioning collision check to use upward normal, useful for 3-axis printing" << std::endl
        << "[-H] : Experimental perimeter hole closing, may cause collisions." << std::endl
        ;return 1;
    }

    float nozzleWidth = 0.6f, layerHeightRateo = 0.5f, topAngle = 30.0f, bottomAngle = 2.0f, numberOfCover = 4.0f, collisionAngle = 95.0f;
    uint32_t zigzagOffset = 0, hardCodedField = 0;
    const char *meshPath = argv[1], *gcode3Path = nullptr, *gcode5Path = nullptr, *plyPath = nullptr;
    bool upCollisionCheck = false, holeClosing = false;
    Infill::Type infill = Infill::Type::Gyroid;
    NormalField::Objective objective = NormalField::Objective::ConformalTop;
    
    for (int i = 1; i < argc-1; ++i) {
		if (!strcmp(argv[i], "-d")) {
            nozzleWidth = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "-h")) {
            layerHeightRateo = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "-t")) {
            topAngle = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "-b")) {
            bottomAngle = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "-3")) {
            gcode3Path = argv[++i];
        } else if (!strcmp(argv[i], "-5")) {
            gcode5Path = argv[++i];
        } else if (!strcmp(argv[i], "-p")) {
            plyPath = argv[++i];
        } else if (!strcmp(argv[i], "-i")) {
            infill = Infill::Type(std::atoi(argv[++i]));
        } else if (!strcmp(argv[i], "-o")) {
            objective = NormalField::Objective(std::atoi(argv[++i]));
        } else if (!strcmp(argv[i], "-c")) {
            numberOfCover = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "-C")) {
            collisionAngle = std::atof(argv[++i]);
        } else if (!strcmp(argv[i], "-z")) {
            zigzagOffset = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-u")) {
            hardCodedField = std::atoi(argv[++i]);
        }
    }

    for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-U")) {
           upCollisionCheck = true;
        } else if (!strcmp(argv[i], "-H")) {
           holeClosing = true;
        }
    }

    std::cout << "Input mesh: " << meshPath << std::endl
        << "Nozzle diameter: " << nozzleWidth << " mm" << std::endl
        << "Layer height: " << layerHeightRateo*nozzleWidth << " mm" << std::endl
        << "Top max angle: " << topAngle << " deg" << std::endl
        << "Bottom max angle: " << bottomAngle << " deg" << std::endl
        << "Infill type: " << Infill::infillToString(infill) << std::endl
        << "Tool orientation: " << NormalField::objectiveToString(objective) << std::endl
        << "Zigzag offset: " << zigzagOffset << std::endl
        << "Collision angle: " << collisionAngle << " deg" << std::endl;
        if (hardCodedField) std::cout << "Hard-coded fields: Enabled" << std::endl;
        if (upCollisionCheck) std::cout << "Force partitioning collision check to use upward normal: Enabled" << std::endl;
        if (holeClosing) std::cout << "Experimental perimeter hole closing: Enabled " << std::endl;
        if (objective == NormalField::Objective::ConformalTop) std::cout << "Number of covers: " << numberOfCover << std::endl;
        if (gcode3Path) std::cout << "3 Axis GCode: " << gcode3Path << std::endl;
        if (gcode5Path) std::cout << "5 Axis GCode: " << gcode5Path << std::endl;

    auto startTimeTotal = std::chrono::steady_clock::now();
    auto startTime = std::chrono::steady_clock::now();
    std::cout << "\nSTL Loading..." << std::endl;
    StlLoader mesh(meshPath);
    printProgress(1.0f);
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count()/1000.0 << " s" << std::endl;

    Slicer slicer(mesh.getTriangleSoup(), nozzleWidth, layerHeightRateo*nozzleWidth, topAngle, bottomAngle, infill, objective, hardCodedField, numberOfCover, zigzagOffset, collisionAngle, upCollisionCheck, holeClosing);
    if (plyPath) slicer.saveToPLY(plyPath);
    if (gcode3Path) slicer.saveToGCode3Axis(gcode3Path);
    if (gcode5Path) slicer.saveToGCode(gcode5Path);

    std::cout << "\nTotal time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTimeTotal).count()/1000.0 << " s" << std::endl;

    return 0;
}