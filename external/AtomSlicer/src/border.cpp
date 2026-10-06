// MIT License
// Copyright 2026 Giovanni Cocco and Inria

#include "border.h"
#include <map>
#include <array>
#include <vector>

std::set<std::array<uint32_t, 2>> Border::borderEdges(const uint32_t *triangles, uint32_t triangleCounts) {
    std::set<std::array<uint32_t, 2>> edgeCuts;
    uint32_t nfaces = triangleCounts;
    if (nfaces == 0) {
        return edgeCuts;
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t> > edges;
    for (uint32_t i = 0; i < nfaces; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            uint32_t v0 = triangles[3*i+j];       
            uint32_t v1 = triangles[3*i+(j + 1)%3];
            std::pair<uint32_t, uint32_t> e;
            e.first = std::min(v0, v1);
            e.second = std::max(v0, v1);
            edges[e].push_back(i);
        }
    }
    uint32_t nedges = edges.size();
    std::vector<uint32_t> ev(2*nedges, uint32_t(-1));
    std::vector<uint32_t> ef(2*nedges, uint32_t(-1));
    std::set<uint32_t> boundaryEdges;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edgeidx;
    uint32_t idx = 0;
    for (auto it : edges) {
        edgeidx[it.first] = idx;
        ev[2*idx+0] = it.first.first;
        ev[2*idx+1] = it.first.second;
        ef[2*idx+0] = it.second[0];
        if (it.second.size() > 1) {
            ef[2*idx+1] = it.second[1];
        } else {
            ef[2*idx+1] = uint32_t(-1);
            boundaryEdges.insert(idx);
        }
        idx++;
    }

    for (uint32_t i : boundaryEdges) {
        edgeCuts.insert({ std::min(ev[2*i+0], ev[2*i+1]), std::max(ev[2*i+0], ev[2*i+1]) });
    }
    return edgeCuts;
}