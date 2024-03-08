% Generazione matrice randomica
A = randi([1, 10], 3, 4);
[m,n] = size(A);
k = 2;

% Decomposizione SVD
if m >= n
    % Ricavo degli autovettori e autovalori di A * A' (m x m)
    [U, lambda] = eig(A * A');

    % Ordinamento degli autovalori
    [lambda, indici] = sort(diag(lambda), "descend");
    % Ordinamento delle colonne rispetto agli autovalori
    U = U(:, indici);

    % Trovo i valori singolari
    Sigma = sqrt(diag(lambda));
    % Seleziono le prime k righe e le prime k colonne
    Sigma_k = Sigma(1:k, 1:k);

    % Normalizzazione degli autovettori di U
    for i = 1:m
        U(:, i) = U(:, i) / norm(U(:, i));
    end
    % Selezione delle prime k colonne di U
    U_k = U(:, 1:k);

    % Calcolo V
    V_k = A' * U_k;
    % Normalizzazione degli autovettori di V
    for i = 1:k
        V_k(:, i) = V_k(:, i) / norm(V_k(:, i));
    end
    
else
    % Ricavo degli autovettori e autovalori di A' * A (n x n)
    [V, lambda] = eig(A' * A);

    % Ordinamento degli autovalori
    [lambda, indici] = sort(diag(lambda), "descend");
    % Ordinamento delle colonne rispetto agli autovalori
    V = V(:, indici);

    % Trovo i valori singolari
    Sigma = sqrt(diag(lambda));
    % Seleziono le prime k righe e le prime k colonne
    Sigma_k = Sigma(1:k, 1:k);

    % Normalizzazione degli autovettori di V
    for i = 1:n
        V(:, i) = V(:, i) / norm(V(:, i));
    end
    % Selezione delle prime k colonne di U
    V_k = V(:, 1:k);

    % Calcolo U
    U_k = A * V_k;
    % Normalizzazione degli autovettori di U
    for i = 1:k
        U_k(:, i) = U_k(:, i) / norm(U_k(:, i));
    end
end

A_k = round(U_k * Sigma_k * V_k');
