function A_k = qrEigen(A, k)
   [m, n] = size(A);

   [Q, R] = func_getQR(A);

   V = Q' * eye(m, n);

   if m >= n
        % Ricavo degli autovettori e autovalori di R * R' (m x m)
        [U_R, lambda] = eig(R * R');

        % Ordinamento degli autovalori
        [lambda, indici] = sort(diag(lambda), "descend");
        % Ordinamento delle colonne rispetto agli autovalori
        U_R = U_R(:, indici);
        %V = V(:, indici);

        % Trovo i valori singolari
        Sigma = sqrt(diag(lambda));
        % Seleziono le prime k righe e le prime k colonne
        Sigma = Sigma(1:k, 1:k);

        % Normalizzazione degli autovettori di U
        %for i = 1:m
        %    U_R(:, i) = U_R(:, i) / norm(U_R(:, i));
        %end

        % Selezione delle prime k colonne di U
        U_R = U_R(:, 1:k);
        V = V(:, 1:k);
   else
       % Ricavo degli autovettori e autovalori di R' * R (n x n)
        [V_R, lambda] = eig(R' * R);

        % Ordinamento degli autovalori
        [lambda, indici] = sort(diag(lambda), "descend");
        % Ordinamento delle colonne rispetto agli autovalori
        V_R = V_R(:, indici);
        %V = V(:, indici);

        % Trovo i valori singolari
        Sigma = sqrt(diag(lambda));
        % Seleziono le prime k righe e le prime k colonne
        Sigma = Sigma(1:k, 1:k);

        % Seleziono solo i primi k valori singolari e autovettori associati
        V_R = V_R(:, 1:k);
        V = V(:, 1:k);

        % Calcolo U
        U_R = R * V_R;
        % Normalizzazione degli autovettori di U
        %for i = 1:m
         %   U_R(:, i) = U_R(:, i) / norm(U_R(:, i));
        %end
   end

   U = Q * U_R;
   disp(size(U));
   disp(size(Sigma));
   disp(size(V));
   A_k = U * Sigma * V';
end

