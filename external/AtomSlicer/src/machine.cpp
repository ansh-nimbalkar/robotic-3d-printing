// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "machine.h"
#include <iostream>
#include <fstream>
#include <iomanip>

static constexpr float BALL_2DPOS_0_X = -4.07;
static constexpr float BALL_2DPOS_0_Y = -12.16;
static constexpr float BALL_2DPOS_1_X = 304.93;
static constexpr float BALL_2DPOS_1_Y = -12.16;
static constexpr float BALL_2DPOS_2_X = 150.43;
static constexpr float BALL_2DPOS_2_Y = 296.84;
static constexpr float BALL_POS_Z = -45.7;
static constexpr float RAIL_ANGLE_0 = 29.89;
static constexpr float RAIL_ANGLE_1 = -29.89;
static constexpr float RAIL_ANGLE_2 = 90.0;
static constexpr float Z_OFFSET = 75.0; // Update header and footer too if this change!
static constexpr float MAX_TILT_ANGLE_DEG = 30.0;
static constexpr float MAX_X_AXIS = 300;
static constexpr float MAX_Y_AXIS = 293;
static constexpr float MAX_Z_AXIS = 280;
static constexpr float BALL_TO_CORNER = 10;
static constexpr float NOZZLE_TO_GAUNTRY = 70;
static constexpr float FILAMENT_DIAMETER = 1.75;
static constexpr float DEPOSITON_FEEDRATE = 600;
static constexpr float TRAVEL_FEEDRATE = 3000;
static constexpr float Z_FAN_ON = 2.0;
static constexpr float RETRACT_THRESH = 1.8;
static constexpr float RETRACT_LENGTH = 2.0;
static constexpr float RETRACT_SPEED = 2700;

const char *HEADER_3AXIS = 
R"(G21 ; set units to millimeters
G90 ; use absolute coordinates
M82 ; use absolute distances for extrusion
M190 S55 ; wait for bed temperature to be reached
M104 S210 ; set temperature
G28 ; home all axes
G0 F6200 X0 Y0
BED_MESH_PROFILE LOAD="default"
M109 S210 ; wait for extruder temperature to be reached 
M106 S0 ; fan off
; purging line
T0
G92 E0
G1 Z1.3 F500 ; move z up little to prevent scratching of surface
G1 X0.1 Y20 Z0.3 F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z0.3 F1000.0 E15 ; draw 1st line
G1 X0.4 Y200.0 Z0.3 F1000.0 ; move to side a little
G1 X0.4 Y20 Z0.3 F1000.0 E30 ; draw 2nd line
G1 E28.0 F2700 ; retract
; done purging extruder
M83 ; relative extrusion
)";

const char *FOOTER_3AXIS = 
R"(M82 ; absolute extrusion
G92 E0
M107       ; fan off
; turn off heaters
M104 S0
M140 S0
M107
G28 X Y       ; move back X and Y to origin
G0 F6200 Y280 ; present bed for part removal
M84           ; disable motors
)";

const char *HEADER = 
R"(G21 ; set units to millimeters
G90 ; use absolute coordinates
M190 S55 ; wait for bed temperature to be reached
G32 ; homing and bed calibration
M104 S210 ; set temperature
M109 S210 ; wait for temperature to be reached
T0
M82 ; use absolute distances for extrusion
; switch to enable 3Z mode
M98 P"/macros/enable3Z.g"
M400 ; wait 
; purging line
G92 E0
G1 Z76.3 U76.3 V76.3 F500 ; move z up little to prevent scratching of surface
G1 X0.1 Y20 Z75.3 U75.3 V75.3 F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z75.3 U75.3 V75.3 F1000.0 E15 ; draw 1st line
G1 X0.4 Y200.0 Z75.3 U75.3 V75.3 F1000.0 ; move to side a little
G1 X0.4 Y20 Z75.3 U75.3 V75.3 F1000.0 E30 ; draw 2nd line
G1 E28.0 F2700 ; retract
; done purging extruder
M83 ; relative extrusion
)";

const char *FOOTER = 
R"(M82 ; absolute extrusion"
G92 E0
G1 E-2.0 F2700 ; retract
G92 E0
M104 S0 ; turn off temperature
M140 S0
M106 S0    ; fan off
; switch to disable 3Z mode
M98 P"/macros/disable3Z.g"
M400 ; wait 
)";

typedef std::array<float, 2> Arr2;
typedef std::array<float, 3> Arr3;
typedef std::array<float, 4> Arr4;
typedef std::array<Arr2, 2> Mat22;
typedef std::array<Arr3, 3> Mat33;

static inline float dot(const Arr2 &a, const Arr2 &b) {
    return a[0]*b[0]+a[1]*b[1];
}

static inline float dot(const Arr3 &a, const Arr3 &b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

static inline Arr2 operator-(const Arr2 &a, const Arr2 &b) {
    return Arr2{a[0]-b[0], a[1]-b[1]};
}

static inline Arr2 operator+(const Arr2 &a, const Arr2 &b) {
    return Arr2{a[0]+b[0], a[1]+b[1]};
}

static inline Arr3 operator-(const Arr3 &a, const Arr3 &b) {
    return Arr3{a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

static inline Arr3 operator+(const Arr3 &a, const Arr3 &b) {
    return Arr3{a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}

static inline Arr2 operator*(float a, const Arr2 &b) {
    return Arr2{a*b[0], a*b[1]};
}

static inline Arr3 operator*(float a, const Arr3 &b) {
    return Arr3{a*b[0], a*b[1], a*b[2]};
}

static inline Arr3 normalize(const Arr3 &a) {
    float l = sqrt(dot(a, a));
    return Arr3{a[0]/l, a[1]/l, a[2]/l};
}

static inline Arr3 cross(const Arr3 &a, const Arr3 &b) {
    return Arr3{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

static inline Arr3 mul(const Mat33 &a, const Arr3 &b) {
    return Arr3{dot(a[0], b), dot(a[1], b), dot(a[2], b)};
}

static inline Arr2 mul(const Mat22 &a, const Arr2 &b) {
    return Arr2{dot(a[0], b), dot(a[1], b)};
}

static inline Mat33 transpose(const Mat33 &a) {
    return Mat33{Arr3{a[0][0],a[1][0],a[2][0]}, Arr3{a[0][1],a[1][1],a[2][1]}, Arr3{a[0][2],a[1][2],a[2][2]}};
}

static std::array<float, 6> inverse(Vec3 toolPosition, Vec3 orientation) {
    const Arr2 ball_0_position{BALL_2DPOS_0_X, BALL_2DPOS_0_Y};
    const Arr2 ball_1_position{BALL_2DPOS_1_X, BALL_2DPOS_1_Y};
    const Arr2 ball_2_position{BALL_2DPOS_2_X, BALL_2DPOS_2_Y};
    const Arr2 slot_0_normal{-sinf(RAIL_ANGLE_0 / 180 * M_PI), cosf(RAIL_ANGLE_0 / 180 * M_PI)};
    const Arr2 slot_1_normal{-sinf(RAIL_ANGLE_1 / 180 * M_PI), cosf(RAIL_ANGLE_1 / 180 * M_PI)};
    const Arr2 slot_2_normal{-sinf(RAIL_ANGLE_2 / 180 * M_PI), cosf(RAIL_ANGLE_2 / 180 * M_PI)};
    const Arr4 constraint_0{slot_0_normal[0], slot_0_normal[1], 0, 0};
    const Arr4 constraint_1{slot_1_normal[0], slot_1_normal[1], 0, -dot(ball_1_position - ball_0_position, slot_1_normal)};
    const Arr4 constraint_2{slot_2_normal[0], slot_2_normal[1], 0, -dot(ball_2_position - ball_0_position, slot_2_normal)};
    const Arr2 l1{ball_1_position - ball_0_position};
	const Arr2 l2{ball_2_position - ball_0_position};
    const Arr2 l1_rot{-l1[1], l1[0]};
    const Arr2 l2_rot{-l2[1], l2[0]};
    const Arr3 ball_0_3d_position{ball_0_position[0], ball_0_position[1], BALL_POS_Z};
    const float z_offset = Z_OFFSET;

    float offset = 0.0f;
    if (toolPosition.z < 0)
        offset = std::numeric_limits<float>::infinity();

    if (acos(orientation.z)/M_PI*180 > MAX_TILT_ANGLE_DEG + 1e-4)
        offset = std::numeric_limits<float>::infinity();
    
    Arr3 normal{-orientation[0], -orientation[1], orientation[2]};
    normal = normalize(normal);
    const Arr3 bitanget = normalize(cross(normal, Arr3{1, 0, 0}));
    const Arr3 tangent = normalize(cross(bitanget, normal));
	const Mat33 TBN{tangent, bitanget, normal};

    Arr3 tmp;
    tmp = mul(TBN, Arr3{constraint_0[0], constraint_0[1], constraint_0[2]});
	const Arr2 n0{tmp[0], tmp[1]};
    tmp = mul(TBN, Arr3{constraint_1[0], constraint_1[1], constraint_1[2]});
	const Arr2 n1{tmp[0], tmp[1]};
    tmp = mul(TBN, Arr3{constraint_2[0], constraint_2[1], constraint_2[2]});
	const Arr2 n2{tmp[0], tmp[1]};
    const float k1 = constraint_1[3];
	const float k2 = constraint_2[3];

    const Arr2 d{-n0[1], n0[0]};
	const float d_n1 = dot(d, n1);
    const float d_n2 = dot(d, n2);
    const float l1_n1 = dot(l1, n1);
    const float l1_rot_n1 = dot(l1_rot, n1);
    const float A = d_n1*dot(l2, n2) - d_n2*l1_n1;
    const float B = d_n1*dot(l2_rot, n2) - d_n2*l1_rot_n1;
    const float C = dot(k2*n1 - k1*n2, d);
    const float theta = A + C == 0 ? 0 :  2*atan2((C + A)/(-B - (B >= 0 ? 1 : -1)*sqrt(B*B + (A + C)*(A - C))), 1.0);
    const float ct = cos(theta);
	const float st = sin(theta);
	const float t = -(l1_n1*ct + l1_rot_n1*st + k1)/d_n1;

    const Mat22 R{Arr2{ct, -st}, Arr2{st, ct}};
    const Mat33 R_3d{Arr3{R[0][0], R[0][1], 0}, Arr3{R[1][0], R[1][1], 0}, Arr3{0, 0, 1}};

	const Arr2 b0_2d = t*d;
    const Arr2 b1_2d = b0_2d + mul(R, l1);
    const Arr2 b2_2d = b0_2d + mul(R, l2);
	const Mat33 TBN_T = transpose(TBN);
	const Arr3 b0 = mul(TBN_T, Arr3{b0_2d[0], b0_2d[1], 0});
    const Arr3 Z{TBN_T[2][0], TBN_T[2][1], TBN_T[2][2]};
	const float b1_z = dot(Z, Arr3{b1_2d[0], b1_2d[1], 0});
	const float b2_z = dot(Z, Arr3{b2_2d[0], b2_2d[1], 0});
	const float delta_z1 = b1_z - b0[2];
	const float delta_z2 = b2_z - b0[2];

    const Arr3 position{toolPosition[0], toolPosition[1], toolPosition[2]};
    const Arr3 pos = mul(TBN_T, mul(R_3d, (position - ball_0_3d_position))) + ball_0_3d_position;
    const float x = pos[0] + b0[0], y = pos[1] + b0[1], z0 = pos[2]+z_offset, z1 = pos[2]-delta_z1+z_offset, z2 = pos[2]-delta_z2+z_offset;

	const float minZ = std::min(std::min(z0, z1), z2);
	if (minZ < 0 && !std::isinf(offset))
		offset = -minZ;

    if (x < 0 || x > MAX_X_AXIS || y < 0 || y > MAX_Y_AXIS || z0 > MAX_Z_AXIS || z1 > MAX_Z_AXIS || z2 > MAX_Z_AXIS)
		offset = std::numeric_limits<float>::infinity();

    Arr3 corner0{ball_0_position[0]-BALL_TO_CORNER, ball_0_position[1]-BALL_TO_CORNER, 0};
	Arr3 corner1{ball_1_position[0]+BALL_TO_CORNER, ball_1_position[1]-BALL_TO_CORNER, 0};
	Arr3 corner2{ball_0_position[0]-BALL_TO_CORNER, ball_2_position[1]+BALL_TO_CORNER, 0};
	Arr3 corner3{ball_1_position[0]+BALL_TO_CORNER, ball_2_position[1]+BALL_TO_CORNER, 0};
	corner0 = mul(TBN_T, mul(R_3d, (corner0 - ball_0_3d_position))) + ball_0_3d_position + Arr3{b0[0], b0[1], -pos[2]};
	corner1 = mul(TBN_T, mul(R_3d, (corner1 - ball_0_3d_position))) + ball_0_3d_position + Arr3{b0[0], b0[1], -pos[2]};
	corner2 = mul(TBN_T, mul(R_3d, (corner2 - ball_0_3d_position))) + ball_0_3d_position + Arr3{b0[0], b0[1], -pos[2]};
	corner3 = mul(TBN_T, mul(R_3d, (corner3 - ball_0_3d_position))) + ball_0_3d_position + Arr3{b0[0], b0[1], -pos[2]};

    const float maxCorner = std::max(std::max(corner0[2], corner1[2]), std::max(corner2[2], corner3[2]));
	if (maxCorner > NOZZLE_TO_GAUNTRY && !std::isinf(offset) && offset < maxCorner - NOZZLE_TO_GAUNTRY)
		offset = maxCorner - NOZZLE_TO_GAUNTRY;
    
    return std::array<float, 6>{x, y, z0, z1, z2, offset};
}

static void moveToolpathXY(std::vector<Toolpath::WayPoint> &toolpath, const Arr2 &offset) {
    for (auto &point : toolpath) {
        point.position.x += offset[0];
        point.position.y += offset[1];
    }
}

static void moveToolpathZ(std::vector<Toolpath::WayPoint> &toolpath, float offset) {
    for (auto &point : toolpath) {
        point.position.z += offset;
    }
}

static float getLift(const std::vector<Toolpath::WayPoint> &toolpath, float width, float height) {
    float zLift = 0;
    float oldLift = 0;
    do {
        oldLift = zLift;
        for (const auto &point : toolpath) {
            auto kinematic = inverse(point.position + Vec3(0, 0, oldLift), point.normal);
            zLift = std::max(zLift, oldLift + kinematic[5]);
            if (std::isinf(zLift)) {
                std::cout << "Fatal Error: the model cannot be placed inside the machine build volume." << std::endl;
                exit(1);
            }
        }
    } while (oldLift != zLift);

    return ceil(zLift/height)*height;
}

void Machine::center3Axis(Toolpath &toolpath, float height) {
    float minX = std::numeric_limits<float>::infinity(), minY = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity(), maxY = -std::numeric_limits<float>::infinity();
    float minXP = std::numeric_limits<float>::infinity(), minYP = std::numeric_limits<float>::infinity();
    float maxXP = -std::numeric_limits<float>::infinity(), maxYP = -std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    for (const auto &point : toolpath.getToolpath()) {
        minX = std::min(minX, point.position.x);
        minY = std::min(minY, point.position.y);
        maxX = std::max(maxX, point.position.x);
        maxY = std::max(maxY, point.position.y);
        minZ = std::min(minZ, point.position.z);
        if (point.position.z < 2*height) {
            minXP = std::min(minXP, point.position.x);
            minYP = std::min(minYP, point.position.y);
            maxXP = std::max(maxXP, point.position.x);
            maxYP = std::max(maxYP, point.position.y);
        }
    }
    Arr2 offsetXY = 0.5f*(Arr2{MAX_X_AXIS, MAX_Y_AXIS} + Arr2{minX, minY}-Arr2{maxX, maxY}) - Arr2{minX, minY};

    float offsetZ = 0.8*height - minZ;
    moveToolpathXY(toolpath.getToolpath(), offsetXY);
    moveToolpathZ(toolpath.getToolpath(), offsetZ);
}

void Machine::addPlatformAndCenter(Toolpath &toolpath, float width, float height) {
    float minX = std::numeric_limits<float>::infinity(), minY = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity(), maxY = -std::numeric_limits<float>::infinity();
    float minXP = std::numeric_limits<float>::infinity(), minYP = std::numeric_limits<float>::infinity();
    float maxXP = -std::numeric_limits<float>::infinity(), maxYP = -std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    for (const auto &point : toolpath.getToolpath()) {
        minX = std::min(minX, point.position.x);
        minY = std::min(minY, point.position.y);
        maxX = std::max(maxX, point.position.x);
        maxY = std::max(maxY, point.position.y);
        minZ = std::min(minZ, point.position.z);
        if (point.position.z < 2*height) {
            minXP = std::min(minXP, point.position.x);
            minYP = std::min(minYP, point.position.y);
            maxXP = std::max(maxXP, point.position.x);
            maxYP = std::max(maxYP, point.position.y);
        }
    }
    Arr2 offsetXY = 0.5f*(Arr2{MAX_X_AXIS, MAX_Y_AXIS} + Arr2{minX, minY}-Arr2{maxX, maxY}) - Arr2{minX, minY};

    float offsetZ = 0.8*height - minZ;
    moveToolpathXY(toolpath.getToolpath(), offsetXY);
    moveToolpathZ(toolpath.getToolpath(), offsetZ);

    float lift = getLift(toolpath.getToolpath(), width, height);
    std::vector<Toolpath::WayPoint> toolpathPlatform;
    toolpathPlatform.reserve(toolpath.getToolpath().size());

    if (lift) {
        std::cout << "Adding a platform of height: " << lift << " mm" << std::endl;
        moveToolpathZ(toolpath.getToolpath(), lift);

        Arr3 size{ float(ceil((maxXP-minXP) / width) * width), float(ceil((maxYP-minYP) / width) * width), lift };
        Vec3 offset(minXP, minYP, 0);
        for (size_t i = 0; i < size_t(size[2] / height) - 1; ++i) {
            float z = (i + 1) * height;

            toolpathPlatform.push_back({ offset+Vec3(0, 0, z), Vec3(0, 0, 1), false });
            toolpathPlatform.push_back({ offset+Vec3(0, size[1], z), Vec3(0, 0, 1), true });
            toolpathPlatform.push_back({ offset+Vec3(size[0], size[1], z), Vec3(0, 0, 1), true });
            toolpathPlatform.push_back({ offset+Vec3(size[0], 0, z), Vec3(0, 0, 1), true });
            toolpathPlatform.push_back({ offset+Vec3(0, 0, z), Vec3(0, 0, 1), true });

            if (i >= size_t(size[2] / height - 3)) {
                float infillPercentage = 0.2;
                size_t direction = 1 - (size_t(size[2] / height - 6) % 2);
                if (i % 2 == direction) {
                    for (size_t j = 1; j < size_t(size[1] / width * infillPercentage); ++j) {
                        float y = j * width / infillPercentage;
                        if (j % 2 == 0) {
                            toolpathPlatform.push_back({ offset+Vec3(0.5 * width, y, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(size[0] - 0.5 * width, y, z), Vec3(0, 0, 1), true });
                        } else {
                            toolpathPlatform.push_back({ offset+Vec3(size[0] - 0.5 * width, y, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(0.5 * width, y, z), Vec3(0, 0, 1), true });
                        }
                    }
                } else {
                    for (size_t j = 1; j < size_t(size[0] / width * infillPercentage); ++j) {
                        float x = j * width / infillPercentage;
                        if (j % 2 == 0) {
                            toolpathPlatform.push_back({ offset+Vec3(x, 0.5 * width, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(x, size[1] - 0.5 * width, z), Vec3(0, 0, 1), true });
                        } else {
                            toolpathPlatform.push_back({ offset+Vec3(x, size[1] - 0.5 * width, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(x, 0.5 * width, z), Vec3(0, 0, 1), true });
                        }
                    }
                }
            } else if (i < 3 || i >= size_t(size[2] / height - 6)) {
                size_t direction = 1 - (size_t(size[2] / height - 6) % 2);
                if (i % 2 == direction) {
                    for (size_t j = 1; j < size_t(size[1] / width); ++j) {
                        float y = j * width;
                        if (j % 2 == 0) {
                            toolpathPlatform.push_back({ offset+Vec3(0.5 * width, y, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(size[0] - 0.5 * width, y, z), Vec3(0, 0, 1), true });
                        } else {
                            toolpathPlatform.push_back({ offset+Vec3(size[0] - 0.5 * width, y, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(0.5 * width, y, z), Vec3(0, 0, 1), true });
                        }
                    }
                } else {
                    for (size_t j = 1; j < size_t(size[0] / width); ++j) {
                        float x = j * width;
                        if (j % 2 == 0) {
                            toolpathPlatform.push_back({ offset+Vec3(x, 0.5 * width, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(x, size[1] - 0.5 * width, z), Vec3(0, 0, 1), true });
                        } else {
                            toolpathPlatform.push_back({ offset+Vec3(x, size[1] - 0.5 * width, z), Vec3(0, 0, 1), false });
                            toolpathPlatform.push_back({ offset+Vec3(x, 0.5 * width, z), Vec3(0, 0, 1), true });
                        }
                    }
                }
            } else {
                float infillPercentage = 0.2;
                for (size_t j = 1; j < size_t(size[1] / width * infillPercentage); ++j) {
                    float y = j * width / infillPercentage;
                    if (j % 2 == 0) {
                        toolpathPlatform.push_back({ offset+Vec3(0.5 * width, y, z), Vec3(0, 0, 1), false });
                        toolpathPlatform.push_back({ offset+Vec3(size[0] - 0.5 * width, y, z), Vec3(0, 0, 1), true });
                    } else {
                        toolpathPlatform.push_back({ offset+Vec3(size[0] - 0.5 * width, y, z), Vec3(0, 0, 1), false });
                        toolpathPlatform.push_back({ offset+Vec3(0.5 * width, y, z), Vec3(0, 0, 1), true });
                    }
                }
            }
        }

        moveToolpathXY(toolpathPlatform, offsetXY);
    }
    for (const auto &p : toolpath.getToolpath()) {
        toolpathPlatform.push_back(p);
    }
    toolpath.getToolpath() = std::move(toolpathPlatform);
}

void Machine::toolpathToGCode(const Toolpath &toolpath, const char *path, float width, float height) {
    const auto &waypoints = toolpath.getToolpath();
    std::ofstream gcode(path);
    gcode << std::fixed;
    
    gcode << HEADER;
    bool isFanOn = false, needPrime = true;
    for (size_t i = 0; i < waypoints.size(); ++i) {
        printProgress(i/float(waypoints.size()-1));

        if (needPrime && waypoints[i].deposition) {
            gcode << "G1 E" << std::setprecision(2) << RETRACT_LENGTH << " F" << std::setprecision(0) << RETRACT_SPEED << " ; prime\n";
            needPrime = false;
        } else if (i > 0 && !needPrime && !waypoints[i].deposition) {
            size_t j = i;
            float dist = 0;
            while (j < waypoints.size() && !waypoints[j].deposition) {
                dist += (waypoints[j].position-waypoints[j-1].position).length();
                if (dist > RETRACT_THRESH) {
                    gcode << "G1 E" << std::setprecision(2) << -RETRACT_LENGTH << " F" << std::setprecision(0) << RETRACT_SPEED << " ; retract\n";
                    needPrime = true;
                    break;
                }
                ++j;
            }
        }

        auto a = inverse(waypoints[i].position, waypoints[i].normal);
        float e = 0, f = TRAVEL_FEEDRATE;
        if (i > 0 && waypoints[i].deposition) {
            float crossSection = 0.25f*M_PI*FILAMENT_DIAMETER*FILAMENT_DIAMETER;
            float distance = (waypoints[i].position-waypoints[i-1].position).length();
            e = distance*width*height/crossSection;

            auto aP = inverse(waypoints[i-1].position, waypoints[i-1].normal);
            float x2 = (a[0]-aP[0])*(a[0]-aP[0]), y2 = (a[1]-aP[1])*(a[1]-aP[1]), z2 = (a[2]-aP[2])*(a[2]-aP[2]),
                    u2 = (a[3]-aP[3])*(a[3]-aP[3]), v2 = (a[4]-aP[4])*(a[4]-aP[4]), e2 = e*e;
            float gcodeDist = sqrt(x2+y2+z2+u2+v2+e2);
            f = DEPOSITON_FEEDRATE * (distance < std::numeric_limits<float>::min() ? 1.0f : gcodeDist/distance);
        }
        gcode << "G1 X" << std::setprecision(6) << a[0] << " Y" << a[1] << " Z" << a[2] << " U" << a[3] << " V" << a[4]
                << " E" << e << " F" << std::setprecision(0) << f << "\n";

        if (isFanOn && waypoints[i].position.z < Z_FAN_ON) {
            gcode << "M106 S0 ; fan off\n";
            isFanOn = false;
        } else if (!isFanOn && waypoints[i].position.z > Z_FAN_ON) {
            gcode << "M106 S255 ; fan on\n";
            isFanOn = true;
        }
    }
    gcode << FOOTER;
}

void Machine::toolpathToGCode3Axis(const Toolpath &toolpath, const char *path, float width, float height) {
    const auto &waypoints = toolpath.getToolpath();
    std::ofstream gcode(path);
    gcode << std::fixed;
    
    gcode << HEADER_3AXIS;
    bool isFanOn = false, needPrime = true;
    for (size_t i = 0; i < waypoints.size(); ++i) {
        printProgress(i/float(waypoints.size()-1));

        if (needPrime && waypoints[i].deposition) {
            gcode << "G1 E" << std::setprecision(2) << RETRACT_LENGTH << " F" << std::setprecision(0) << RETRACT_SPEED << " ; prime\n";
            needPrime = false;
        } else if (i > 0 && !needPrime && !waypoints[i].deposition) {
            size_t j = i;
            float dist = 0;
            while (j < waypoints.size() && !waypoints[j].deposition) {
                dist += (waypoints[j].position-waypoints[j-1].position).length();
                if (dist > RETRACT_THRESH) {
                    gcode << "G1 E" << std::setprecision(2) << -RETRACT_LENGTH << " F" << std::setprecision(0) << RETRACT_SPEED << " ; retract\n";
                    needPrime = true;
                    break;
                }
                ++j;
            }
        }

        float e = 0, f = TRAVEL_FEEDRATE;
        if (i > 0 && waypoints[i].deposition) {
            float crossSection = 0.25f*M_PI*FILAMENT_DIAMETER*FILAMENT_DIAMETER;
            float distance = (waypoints[i].position-waypoints[i-1].position).length();
            e = distance*width*height/crossSection;
            f = DEPOSITON_FEEDRATE;
        }
        gcode << "G1 X" << std::setprecision(6) << waypoints[i].position.x << " Y" << waypoints[i].position.y << " Z" << waypoints[i].position.z <<
                " E" << e << " F" << std::setprecision(0) << f << " ; " << std::setprecision(6) << waypoints[i].normal << "\n";

        if (isFanOn && waypoints[i].position.z < Z_FAN_ON) {
            gcode << "M106 S0 ; fan off\n";
            isFanOn = false;
        } else if (!isFanOn && waypoints[i].position.z > Z_FAN_ON) {
            gcode << "M106 S255 ; fan on\n";
            isFanOn = true;
        }
    }
    gcode << FOOTER_3AXIS;
}