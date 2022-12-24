
mutable struct Path
    segments
    conflicts
    length

    function Path(segs)
        obj = new()
        obj.segments = segs
        obj.conflicts = Dict("with_path" => [], "p_from_entry" => [])
        return obj
    end
end

"Compute the length of path"
function CZ_length(p::Path, MYMAP::Dict)
    p.length = sum([MYMAP[s]["length"] for s in p.segments])
end

"Add conflict points to the path"
function add_conflicts(p::Path, MYMAP::Dict, with_path, on_seg, dist)
    append!(p.conflicts["with_path"], with_path)
    e1 = 0
    for s in p.segments
        if s != on_seg
            seg = MYMAP[s]
            e1 += seg["length"]
        elseif s == on_seg
            e1 += dist; break
        end
    end
    append!(p.conflicts["p_from_entry"], e1)
end

function sort_conflict(p::Path)
    n_conf = length(p.conflicts["p_from_entry"])
    for k in 1:n_conf-1
        if p.conflicts["p_from_entry"][k] > p.conflicts["p_from_entry"][k+1]
            p.conflicts["p_from_entry"][k], p.conflicts["p_from_entry"][k+1] = p.conflicts["p_from_entry"][k+1], p.conflicts["p_from_entry"][k]
            p.conflicts["with_path"][k], p.conflicts["with_path"][k+1] = p.conflicts["with_path"][k+1], p.conflicts["with_path"][k]
        end
    end
end
