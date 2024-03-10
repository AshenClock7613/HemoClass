function [Q,R] = func_getQR(A)
R = A;
[m, n] = size(A);

c = min(m, n);
% Memorizzo tutte le matrici di householder all'interno di un array
H = cell(1, c);

% Applicazione della fattorizzazione QR
for k = 1:c
    x = R(k:m, k); % Corrisponde ad a_k
    e = [1; zeros(length(x)-1, 1)]; % Vettore nullo eccetto prima componente
    sigma = norm(x); %Calcolo della norma 2 di x
    % sign(x(1)) per ottimizzare il risultato
    vk = sign(x(1)) * sqrt(sigma * (sigma + abs(x(1)))) * e + x;

    % Calcolo della matrice di householder H_k
    %hk = eye(length(x)) - 2 * (vk * vk') / (sigma * (sigma + abs(x(1))));
    hk = eye(length(x)) - 2 * (vk * vk') / (vk' * vk);
    if k > 1
        hk = blkdiag(eye(k-1), hk);
    end

    H{k} = hk;
    % Annullo gli elementi sotto la diagonale della colonna k di R
    R = hk * R;
end

% Inizializzazione di Q con la prima matrice H
Q = H{1};
for i = 2:c
    % Calcolo Q moltiplicando riga per colonna tutte le H
    Q = H{i} * Q;
end
disp(size(A));
disp(size(Q));
disp(size(R));
end

