y = func_imread('Immagini/');
compression_ratio = 0.8; % 80% di compressione
A_k_QR = func_householderQR(double(y{1,1}), compression_ratio);
A_k_SVD = func_ourSVD(double(y{1,1}), 50);
autovalori = qrEigen2(double(y{1,1}), 150);
% Conversione double uint8
figure;
imshow(y{1,1});
figure;
imshow(uint8(A_k_QR));
figure;
imshow(uint8(autovalori));

