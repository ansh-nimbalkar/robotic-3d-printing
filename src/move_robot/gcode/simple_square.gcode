; Simple test G-code for TCP pose generation
; Units are in mm

G90
M82
G92 E0

; Move to start height
G0 X0 Y0 Z0.2 F3000

; First square extrusion path
G1 X40 Y0 E1.0 F1200
G1 X40 Y40 E2.0
G1 X0 Y40 E3.0
G1 X0 Y0 E4.0

; Travel move to second shape
G0 X60 Y0 F3000

; Second extrusion path
G1 X80 Y0 E5.0 F1200
G1 X80 Y20 E6.0
G1 X60 Y20 E7.0
G1 X60 Y0 E8.0