import bpy

print("\n" + "="*60)
print("aluminum-specular-mat material nodes")
print("="*60)

mat = bpy.data.materials.get('aluminum-specular-mat')
if mat is None:
    print("Material not found")
else:
    nt = mat.node_tree
    print(f"\nNodes ({len(nt.nodes)}):")
    for n in nt.nodes:
        print(f"  [{n.type}] {n.name}")
        for inp in n.inputs:
            try:
                default = inp.default_value
                if hasattr(default, '__len__'):
                    default = list(default)
                print(f"    in:  {inp.name:20s} type={inp.type:10s} default={default} linked={inp.is_linked}")
            except Exception:
                print(f"    in:  {inp.name:20s} type={inp.type:10s} (skip)")

    print(f"\nLinks ({len(nt.links)}):")
    for link in nt.links:
        print(f"  {link.from_node.name}.{link.from_socket.name}  ->  {link.to_node.name}.{link.to_socket.name}")

print("\n" + "="*60)
print("GelSurface material slots")
print("="*60)
gel = bpy.data.objects.get('GelSurface')
if gel:
    for slot in gel.material_slots:
        print(f"  slot: {slot.material.name if slot.material else None}")
