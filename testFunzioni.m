y = func_imread('Immagini/');
view_images(y)
 compression_ratio = 0.8; % 80% di compressione
A_k_QR = func_householderQR(y{1}, compression_ratio);
A_k_SVD = func_ourSVD(y{1}, 2);