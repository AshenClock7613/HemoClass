function images = func_imread(path)
    % Ottieni la lista dei file JPEG nella cartella specificata
    files = dir(fullfile(path, '*.jpeg'));
    
    % Conta il numero di file JPEG nella cartella
    num_jpg_files = length(files);
    
    % Preallocazione della cella per memorizzare le immagini
    images = cell(1, num_jpg_files);
    
    % Scansiona tutti i file JPEG e carica le immagini
    for i = 1:num_jpg_files
        filename = fullfile(path, files(i).name); % Costruisci il percorso completo del file
        RGB = imread(filename); % Leggi l'immagine RGB
        
        % Converti l'immagine RGB in scala di grigi
        gray_image = rgb2gray(RGB);
        
        % Normalizza i valori della matrice nell'intervallo [0, 1]
        images{i} = gray_image; 
    end
end
