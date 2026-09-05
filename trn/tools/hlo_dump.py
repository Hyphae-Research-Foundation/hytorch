import sys
from torch_neuronx.pyhlo.service import hlo_pb2
from torch_neuronx.pyhlo import xla_data_pb2 as xd
def shp(s):
    if s.element_type == xd.TUPLE:
        return "tuple(%s)" % ",".join(shp(t) for t in s.tuple_shapes)
    return f"{xd.PrimitiveType.Name(s.element_type).lower()}[{','.join(map(str,s.dimensions))}]"
def dump(path, out_path):
    m = hlo_pb2.HloModuleProto(); m.ParseFromString(open(path, "rb").read())
    out = []
    for c in m.computations:
        out.append(f"== computation {c.name}")
        for i in c.instructions:
            extra = ""
            if i.opcode == "custom-call": extra = " target=" + i.custom_call_target
            if i.opcode == "dot":
                d = i.dot_dimension_numbers
                extra = f" lhs_c={list(d.lhs_contracting_dimensions)} rhs_c={list(d.rhs_contracting_dimensions)} lhs_b={list(d.lhs_batch_dimensions)} rhs_b={list(d.rhs_batch_dimensions)}"
            if i.opcode == "gather":
                g = i.gather_dimension_numbers
                extra = f" offset{list(g.offset_dims)} coll{list(g.collapsed_slice_dims)} start{list(g.start_index_map)} slice={list(i.gather_slice_sizes)}"
            if i.opcode == "slice": extra = " " + ",".join(f"[{d.start}:{d.limit}:{d.stride}]" for d in i.slice_dimensions)
            if i.opcode in ("reduce", "broadcast", "concatenate", "iota", "transpose", "reverse"): extra = f" dims={list(i.dimensions)}"
            if i.opcode == "call": extra = f" to={list(i.called_computation_ids)}"
            out.append(f"  %{i.id}:{i.name} = {i.opcode} {shp(i.shape)} ops={list(i.operand_ids)}{extra}")
    open(out_path, "w").write("\n".join(out) + "\n")
    print(len(out), "lines ->", out_path)
if __name__ == "__main__":
    dump(sys.argv[1], sys.argv[2])
