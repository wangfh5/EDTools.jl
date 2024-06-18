ordered_pair(i::T, j::T) where {T} = (i ≤ j) ? (i, j) : (j, i)

"""
    npbc(x::T, L::T) where {T<:Integer}
Input a 1D coordinate `x` and a 1D lattice size `L`,
return the periodic boundary condition (PBC) coordinate.
"""
npbc(x::T, L::T) where {T<:Integer} = mod1(x, L)
"""
    npbc(x::NTuple{N,T}, L::NTuple{N,T}) where {N,T<:Integer}
Input a multi-dimensional coordinate `x` and a multi-dimensional lattice size `L`,
return the periodic boundary condition (PBC) coordinate with format `CartesianIndex`.
"""
npbc(x::NTuple{N,T}, L::NTuple{N,T}) where {N,T<:Integer} = CartesianIndex(mod1.(Tuple(x), L))

"""
    function pbcshift(p, s, L)
Input a site `p`, a shift `s`, and the linear sizes of lattice `L`,
return the periodic boundary condition (PBC) shifted site `npbc(p+s)` with format `CartesianIndex`.
"""
@inline function pbcshift(p, s, L)::CartesianIndex
    return mod1.(Tuple(p) .+ Tuple(s), L) |> CartesianIndex
end

"""
    find_nn_list(L::NTuple{N,Int}, nn_vec::Vector{NTuple{N,Int}}; pbc::NTuple{N,Bool}=ntuple(i -> true, N)) where N
Find the list of neighbors at distance `nn_vec` for each site in the `N`-dimensional lattice with size `L`. 
The boundary condition along each dimension is determined by the `pbc` tuple. By default, all dimensions are PBC.
Note that for 2D lattice, `pbc = (true, false)` corresponds to y (different rows) PBC and x (different columns) OBC.
"""
function find_nn_list(L::NTuple{N,Int}, nn_vec::Vector{NTuple{N,Int}}; pbc::NTuple{N,Bool}=ntuple(i -> true, N)) where N
    Lids = LinearIndices(L)
    if all(pbc)
        nn_list = [
            [Lids[pbcshift(p,s,L)] for s ∈ nn_vec]
            for p ∈ CartesianIndices(L)
        ]
    else
        nn_list = [
            [
                let pn = Tuple(p) .+ Tuple(s)
                    if all(map(x -> x[1] || x[2], zip(1 .<= pn .<= L, pbc)))
                        Lids[npbc(pn, L)]
                    else
                        nothing
                    end
                end for s ∈ nn_vec
            ]
            for p ∈ CartesianIndices(L)
        ]
    end
    return nn_list
end