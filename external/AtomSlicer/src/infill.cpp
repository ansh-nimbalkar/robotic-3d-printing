// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "infill.h"
#include "parallel.h"
#include <iostream>

static float myMod(float a, float b) {
    return fmodf(fmodf(a, b)+b, b);
}

void Infill::generate(SDF &sdf, Type type, float depositionWidth) {
    Parallel::ForProgress(0, sdf.getSize(0)*sdf.getSize(1)*sdf.getSize(2), [&sdf, type, depositionWidth](size_t i) {
        float wallWidth = 2.0f*depositionWidth;
        float inputSDF = sdf.getField()[i];
        float hollowSDF = fabsf(inputSDF+0.5f*wallWidth)-0.5f*wallWidth;

        if (type == Type::Hollow) {
            sdf.getField()[i] = hollowSDF;
        } else if (type == Type::Full) {
            sdf.getField()[i] = inputSDF;
        } else if (type == Type::Grid) {
            Vec3 cellCenter = sdf.getPosition(i);
            float gridScale = 10.0f*depositionWidth;
            float fromCenterX = -fabsf(fmodf(cellCenter.x, gridScale)-0.5f*gridScale)+0.5f*gridScale-depositionWidth;
            float fromCenterY = -fabsf(fmodf(cellCenter.y, gridScale)-0.5f*gridScale)+0.5f*gridScale-depositionWidth;
            float infillSDF = std::min(fromCenterX, fromCenterY);
            sdf.getField()[i] = std::min(std::max(inputSDF, infillSDF), hollowSDF);
        } else if (type == Type::Cubic) {
            Mat3 M(0.81915204, 0.40557979, 0.40557979, 0.0, 0.70710678, -0.70710678, -0.57357644, 0.57922797, 0.57922797);
            Vec3 cellCenter = sdf.getPosition(i)*M;
            float gridScale = 10.0f*depositionWidth;
            float fromCenterX = -0.9f*fabsf(myMod(cellCenter.x, gridScale)-0.5f*gridScale)+0.5f*gridScale-depositionWidth;
            float fromCenterY = -0.9f*fabsf(myMod(cellCenter.y, gridScale)-0.5f*gridScale)+0.5f*gridScale-depositionWidth;
            float fromCenterZ = -0.9f*fabsf(myMod(cellCenter.z, gridScale)-0.5f*gridScale)+0.5f*gridScale-depositionWidth;
            float infillSDF = std::min(std::min(fromCenterX, fromCenterY), fromCenterZ);
            sdf.getField()[i] = std::min(std::max(inputSDF, infillSDF), hollowSDF);
        } else if (type == Type::Gyroid) {
            Vec3 q = 0.5f*sdf.getPosition(i)/depositionWidth*0.6f;
            float x = q.x, y = q.y, z = q.z; 
            float f = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x);
            Vec3 g = Vec3(cos(x)*cos(y)-sin(z)*sin(x), cos(y)*cos(z)-sin(x)*sin(y), cos(z)*cos(x)-sin(y)*sin(z));
            float gLen = std::max(1e-4f, g.length());
            float offset = 1.4f*0.5f*depositionWidth;
            float infillSDF = 2.0f*(fabsf(f)-offset)/gLen;
            sdf.getField()[i] = std::min(std::max(inputSDF, infillSDF), hollowSDF);
        }
    });
}

std::string Infill::infillToString(Infill::Type infill) {
    switch (infill) {
    case Full:
        return "Full";
    case Hollow:
        return "Hollow";
    case Grid:
        return "Grid";
    case Cubic:
        return "Cubic";
    case Gyroid:
        return "Gyroid";
    default:
        return "";
    }
}
