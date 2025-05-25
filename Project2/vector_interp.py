def vector_interp(p1, p2, V1, V2, coord, dim):
    idx = dim - 1

    if p1[idx] == p2[idx]:
        return V1

    t = (coord - p1[idx]) / (p2[idx] - p1[idx])
    V = [v1 + t * (v2 - v1) for v1, v2 in zip(V1, V2)]
    return V