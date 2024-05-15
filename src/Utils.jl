export qdcm, skew_symetric_matrix_quat, skew_symetric_matrix

function qdcm(q)
    q_norm = (q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)^0.5
    w, x, y, z = q ./ q_norm
    return [1-2*(y^2+z^2) 2*(x*y-z*w) 2*(x*z+y*w);
        2*(x*y+z*w) 1-2*(x^2+z^2) 2*(y*z-x*w);
        2*(x*z-y*w) 2*(y*z+x*w) 1-2*(x^2+y^2)]
end

function skew_symetric_matrix_quat(w)
    x, y, z = w
    return [0 -x -y -z;
        x 0 z -y;
        y -z 0 x;
        z y -x 0]
end

function skew_symetric_matrix(w)
    x, y, z = w
    return [0 -z y;
        z 0 -x;
        -y x 0]
end