"""Generate the structured Quad4 mesh used by case 10 (Gmsh 2.2 ASCII)."""
from pathlib import Path
import gmsh


def main():
    out=Path(__file__).parent/"assets"/"clamped_square_shell4_v22.msh"
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("clamped_square_shell4")
    p=[gmsh.model.geo.addPoint(x,y,0) for x,y in ((0,0),(1,0),(1,1),(0,1))]
    lines=[gmsh.model.geo.addLine(p[i],p[(i+1)%4]) for i in range(4)]
    loop=gmsh.model.geo.addCurveLoop(lines); surface=gmsh.model.geo.addPlaneSurface([loop])
    for line in lines: gmsh.model.geo.mesh.setTransfiniteCurve(line,9)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface); gmsh.model.geo.mesh.setRecombine(2,surface)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1,lines,name="clamped")
    gmsh.model.addPhysicalGroup(2,[surface],name="plate")
    gmsh.model.mesh.generate(2); gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.option.setNumber("Mesh.Binary",0); gmsh.write(str(out)); gmsh.finalize()
    print(out)


if __name__=="__main__": main()
