% Generazione matrice randomica
A = randi([1, 10], 5, 5);
[Qmatlab, Rmatlab] = qr(A);

% Inizializzazione di R e determinazione delle dimensioni
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

% Troncamento delle matrici Q e R del 80%
compression_ratio = 0.2;
num_col_troncate = round(size(Q, 2) * compression_ratio);
num_row_troncate = round(size(R, 1) * compression_ratio);

Q_troncata = Q(:, 1:num_col_troncate);
R_troncata = R(1:num_row_troncate, :);

% Ricostruzione dell'immagine compressa
A_k = round(Q_troncata * R_troncata);

for i = 1:m
    for j = 1:n
        A_k(i, j) = abs(A_k(i, j));
    end
end
