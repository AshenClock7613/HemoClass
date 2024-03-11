function A_k = qrEigen2(A, k)
    [m, n] = size(A);

    R1 = A;
    U = eye(m);
    V = eye(n);

    for i = 1 : k
        [Q1, R1] = qr(R1);
        [Q2, R2] = qr(R1');
        R1 = R2';

        if m <= n
            U = U * Q1;
            V = V * Q2;
        else
            U = Q1 * U;
            V = Q2 * V;
        end
    end

    S = R1;
    A_k = U * S * V';
end
