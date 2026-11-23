// NAFEMS LE10, thick elliptical plate under pressure, in metres.
// Geometry follows the public Gmsh definition by Seamplex/FeenoX:
// https://www.seamplex.com/fino/cases/012-nafems-le10/
SetFactory("OpenCASCADE");

a = 1.0; b = 2.75; c = 3.25; d = 2.0; h = 0.6;
DefineConstant[lc = {0.30, Name "Parameters/lc"}];
Point(1) = {0, a, 0};
Point(2) = {0, b, 0};
Point(3) = {c, 0, 0};
Point(4) = {d, 0, 0};
Line(1) = {1, 2};
Ellipse(2) = {0, 0, 0, c, b, 0, Pi/2};
Line(3) = {3, 4};
Ellipse(4) = {0, 0, 0, d, a, 0, Pi/2};
Coherence;
Curve Loop(1) = {1, -2, 3, 4};
Plane Surface(1) = {1};
Extrude {0, 0, 0.5*h} { Surface{1}; }
Extrude {0, 0, 0.5*h} { Surface{6}; }

// These names are the benchmark's boundary-condition notation.
Physical Surface("DCDpCp") = {9, 4};       // uy = 0 (y=0 symmetry)
Physical Surface("ABApBp") = {7, 2};       // ux = 0 (x=0 symmetry)
Physical Surface("BCBpCp") = {8, 3};       // ux = uy = 0 (outer ellipse)
Physical Curve("midplane") = {9};          // uz = 0 on outer edge z=h/2
Physical Surface("upper") = {11};          // nominal -z pressure surface
Physical Volume("bulk") = {1, 2};

// Tet10 makes this compatible with the current VFIFE solid-element set.
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.ElementOrder = 2;
Mesh.SecondOrderIncomplete = 0;
