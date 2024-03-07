A = randi([1, 10], 5, 5);
compression_ratio = 0.8; % 80% di compressione
A_k_QR = func_householderQR(A, compression_ratio);
A_k_SVD = func_ourSVD(A, 2);