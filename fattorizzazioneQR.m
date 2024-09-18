function [Q, R] = fattorizzazioneQR(A) 
 
    [m, n] = size(A); 
    %Inizializzazione delle matrici 
    Q = eye(m);  
    R = A; 
 
    for i = 1 : n 
 
        %Calcolo Householder 
 
        x = R(i:m, i); 
        e = zeros(length(x), 1); 
 
        e(1) = 1; %imposto ad 1 il primo elemento 
 
        v = sign(x(1)) * norm(x) * e + x; 
        % il segno del primo elemento di x * norma aggiungendo x al 
        % risultato, v punta in direzione opposta di v ma con la stessa 
        % norma 
 
        v = v / norm(v); 
        
        % aggiorna la sottomatrice di R 
        R(i:m, i:n) = R(i:m, i:n) -2 * v * (v' * R(i:m, i:n));
 
        %aggiorna la sottomatrice di Q
        Q(i:m, :) = Q(i:m, :) -2 * v * (v' * Q(i:m, :));
    end 
end