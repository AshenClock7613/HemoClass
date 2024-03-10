function [eigenvalue,eigenvector] = qrEigen(A, maxIter)
[m, n] = size(A);
M = A;
% Il numero di iterazioni non deve superare il numero di righe e colonne
nIter = min(maxIter, max(m, n));
% Inizializzazione eigenvector
eigenvector = eye(m);

% Metodo QR iterativo
    for k = 1:nIter
        [Q, R] = func_getQR(M);
        % Aggiornamento di M con sottomatrice RQ
        M = R * Q';
        % Calcolo autovettore
        eigenvector = eigenvector * Q';
    end
    % Estrazione autovalore
    eigenvalue = diag(A);
end

