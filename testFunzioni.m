y = func_imread('Immagini/');
A_k_SVD = func_ourSVD(double(y{1,1}), 50);
autovalori =func_qrSVD(double(y{1,1}), 20);
% Conversione double uint8
figure;
imshow(y{1,1});
figure;
imshow(uint8(A_k_SVD));
figure;
imshow(uint8(autovalori));

