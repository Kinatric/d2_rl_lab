"""Compose the verified robot physics with URDF visuals and bundled MuJoCo scenery."""

import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def visual_scene_xml(physics_path: Path, terrain: str, urdf_name: str) -> str:
    """Add massless, non-colliding meshes without rewriting validated assets."""
    if terrain not in ("flat", "stairs"):
        raise ValueError(f"Unsupported robot visual terrain: {terrain}")
    physics_path = Path(physics_path).resolve()
    root = ET.parse(physics_path).getroot()
    scenery = ET.parse(Path(__file__).resolve().parents[1] / "assets" / f"{terrain}.xml").getroot()
    world = root.find("worldbody")
    for tag in ("statistic", "visual", "asset"):
        root.append(copy.deepcopy(scenery.find(tag)))
    asset = root.find("asset")
    root.find("compiler").set("meshdir", str(physics_path.parent / "meshes"))
    for element in list(world):
        if element.tag == "light" or (terrain == "stairs" and element.tag == "geom"):
            world.remove(element)
    for element in scenery.find("worldbody"):
        if element.tag == "light" or terrain == "stairs":
            world.append(copy.deepcopy(element))
    if terrain == "flat":
        # Keep the original plane's physical properties, only dress its surface.
        floor = world.find("geom[@name='ground']")
        floor.set("material", "groundplane")
        floor.set("rgba", "1 1 1 1")
    urdf = ET.parse(physics_path.parent / "urdf" / urdf_name)
    for link in urdf.findall("link"):
        body = root.find(f'.//body[@name="{link.get("name")}"]')
        if body is None:
            continue
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            for geom in body.findall("geom"):
                if geom.get("contype") != "0":
                    geom.set("group", "3")
            name = f'visual_{link.get("name")}_{index}'
            ET.SubElement(asset, "mesh", name=name,
                          file=str(physics_path.parent / "meshes" / Path(mesh.get("filename")).name),
                          scale=mesh.get("scale", "1 1 1"))
            origin = visual.find("origin")
            xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            r, p, y = (float(v) / 2 for v in rpy.split())
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            quat = (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                    cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, pos=xyz,
                          quat=" ".join(map(str, quat)), contype="0", conaffinity="0",
                          mass="0", group="2", rgba="0.65 0.68 0.72 1")
    return ET.tostring(root, encoding="unicode")
