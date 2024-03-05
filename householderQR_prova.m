%generazione matrice randomica
A = randi([1, 10], 5, 5);
[Qmatlab, Rmatlab] = qr(A);

R = A;
[m, n] = size(A);
c = min(m, n);
H = cell(1, c);

for k = 1:c
    x = R(k:m, k); %corrisponde ad a_1
    e = [1; zeros(length(x)-1, 1)]; %vettore nullo eccetto prima componente
    vk = sign(x(1)) * sqrt(sum(x.^2)) * e + x;

    %calcolo H_k
    hk = eye(length(x)) - 2 * (vk * vk') / (vk' * vk);
    if k > 1
        hk = blkdiag(eye(k-1), hk);
    end

    H{k} = hk;
    R = hk * R;
end

%inizializzo Q con la prima matrice H
Q = H{1};
for i = 2:c
    %calcolo Q moltiplicando riga per colonna tutte le H
    Q = H{i} * Q;
end