function A_k = func_ourSVD(A, k)
    [m, n] = size(A);

    % SVD Decomposition
    if m >= n
        % Compute eigenvalues and eigenvectors of A * A'
        [U, D] = eig(A * A');
        eigenvalues = diag(D);

        % Sort eigenvalues and corresponding eigenvectors in descending order
        [eigenvalues_sorted, indices] = sort(eigenvalues, 'descend');
        U = U(:, indices);

        % Compute singular values
        singular_values = sqrt(max(eigenvalues_sorted, 0)); % Ensure non-negative

        % Select the top k singular values and corresponding vectors
        Sigma_k = singular_values(1:k);
        U_k = U(:, 1:k);

        % Avoid division by zero
        non_zero_indices = Sigma_k > eps;
        Sigma_k_non_zero = Sigma_k(non_zero_indices);
        U_k_non_zero = U_k(:, non_zero_indices);

        % Compute V_k
        V_k = zeros(n, sum(non_zero_indices));
        for i = 1:length(Sigma_k_non_zero)
            V_k(:, i) = (A' * U_k_non_zero(:, i)) / Sigma_k_non_zero(i);
        end

        % Normalize V_k vectors
        for i = 1:size(V_k, 2)
            V_k(:, i) = V_k(:, i) / norm(V_k(:, i));
        end

        % Reconstruct A_k
        A_k = U_k_non_zero * diag(Sigma_k_non_zero) * V_k';
    else
        % Compute eigenvalues and eigenvectors of A' * A
        [V, D] = eig(A' * A);
        eigenvalues = diag(D);

        % Sort eigenvalues and corresponding eigenvectors in descending order
        [eigenvalues_sorted, indices] = sort(eigenvalues, 'descend');
        V = V(:, indices);

        % Compute singular values
        singular_values = sqrt(max(eigenvalues_sorted, 0)); % Ensure non-negative

        % Select the top k singular values and corresponding vectors
        Sigma_k = singular_values(1:k);
        V_k = V(:, 1:k);

        % Avoid division by zero
        non_zero_indices = Sigma_k > eps;
        Sigma_k_non_zero = Sigma_k(non_zero_indices);
        V_k_non_zero = V_k(:, non_zero_indices);

        % Compute U_k
        U_k = zeros(m, sum(non_zero_indices));
        for i = 1:length(Sigma_k_non_zero)
            U_k(:, i) = (A * V_k_non_zero(:, i)) / Sigma_k_non_zero(i);
        end

        % Normalize U_k vectors
        for i = 1:size(U_k, 2)
            U_k(:, i) = U_k(:, i) / norm(U_k(:, i));
        end

        % Reconstruct A_k
        A_k = U_k * diag(Sigma_k_non_zero) * V_k_non_zero';
    end

    % Round the reconstructed matrix
    A_k = round(A_k);
end
