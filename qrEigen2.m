function A_k = qrEigen2(A)
    % Ottieni dimensioni della matrice
    [m, n] = size(A);
    
    % Calcola la fattorizzazione QR di A
    [Q, R] = qr(A);
    
    % Risolvi il sistema Rx = Q^T * b per ottenere V
    V = Q' * eye(m, n);
    
    % Calcola la SVD di R
    [U_R, S, V_R] = svd(R);
    
    % Calcola U
    U = Q * U_R;
    
    % Ordina gli elementi singolari in modo decrescente
    [S, idx] = sort(diag(S), 'descend');
    
    % Riordina le colonne di U e V
    U = U(:, idx);
    V = V(:, idx);
    
    % Costruisci la matrice singolare S
    A_k = U * diag(S) * V';
end
