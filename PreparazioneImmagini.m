% Imposta la directory di partenza e di output
base_dir = 'training_data/';
output_dir = 'training_data_compresso/';

% Crea la directory di output se non esiste
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Ottieni le cartelle 'fold_0', 'fold_1', 'fold_2', ecc.
folds = dir(fullfile(base_dir, 'fold_*'));

% Itera su ciascun fold
for fold_idx = 1:length(folds)
    fold_name = folds(fold_idx).name;
    
    % Ottiene le sottocartelle 'all' e 'hem'
    categories = {'all', 'hem'};
    
    for cat_idx = 1:length(categories)
        category = categories{cat_idx};
        category_dir = fullfile(base_dir, fold_name, category);
        
        % Trova tutte le immagini .bmp nella cartella corrente
        images = func_imread(category_dir);
        
        % Crea la directory di output per questo fold e categoria, mantenendo la stessa struttura
        output_subdir = fullfile(output_dir, fold_name, category);
        if ~exist(output_subdir, 'dir')
            mkdir(output_subdir);
        end
        
        % Itera su ciascuna immagine nella categoria
        for img_idx = 1:length(images)
            % Estrazione delle immagini dalla cella
            img = images{img_idx};
            
            % Applica la decomposizione SVD
            A_k_SVD = func_ourSVD(double(img), 25);

            % Conversione dell'immagine compressa in uint8
            compressed_image = uint8(A_k_SVD);

            % Mantiene lo stesso nome file per l'output, cambiando solo la directory
            output_filename = fullfile(output_subdir, sprintf('image%d_compressed.png', img_idx));
            
            % Salva l'immagine compressa nella cartella di output
            imwrite(compressed_image, output_filename);
        end
    end
end

% CALCOLO DEL RAPPORTO TRA IMMAGINI ORIGINALI E COMPLESSE

% Ottiene le informazioni sui folder fold_0, fold_1, fold_2
fold_info = dir(fullfile(base_dir, 'fold_*'));

% Inizializza la variabile per la somma delle dimensioni
total_original_size = 0;

% Scorre tutti i fold (fold_0, fold_1, fold_2)
for i = 1:length(fold_info)
    % Verifica che sia una directory
    if fold_info(i).isdir
        % Percorso delle sottocartelle 'all' e 'hem' in ciascun fold
        all_dir = fullfile(base_dir, fold_info(i).name, 'all');
        hem_dir = fullfile(base_dir, fold_info(i).name, 'hem');
        
        % Calcola la dimensione dei file nella cartella 'all'
        total_original_size = total_original_size + calculate_folder_size(all_dir);
        
        % Calcola la dimensione dei file nella cartella 'hem'
        total_original_size = total_original_size + calculate_folder_size(hem_dir);
    end
end

% Dimensione in MB per le immagini originali
fprintf('Dimensione totale in MB delle immagini originali: %.2f MB\n', total_original_size / 1e6);

% -----------------------------------------------------------------------

% Ottiene le informazioni sui folder fold_0, fold_1, fold_2
fold_info = dir(fullfile(output_dir, 'fold_*'));

% Inizializza la variabile per la somma delle dimensioni
total_compressed_size = 0;

% Scorre tutti i fold (fold_0, fold_1, fold_2)
for i = 1:length(fold_info)
    % Verifica che sia una directory
    if fold_info(i).isdir
        % Percorso delle sottocartelle 'all' e 'hem' in ciascun fold
        all_dir = fullfile(output_dir, fold_info(i).name, 'all');
        hem_dir = fullfile(output_dir, fold_info(i).name, 'hem');
        
        % Calcola la dimensione dei file nella cartella 'all'
        total_compressed_size = total_compressed_size + calculate_folder_size(all_dir);
        
        % Calcola la dimensione dei file nella cartella 'hem'
        total_compressed_size = total_compressed_size + calculate_folder_size(hem_dir);
    end
end

% Dimensione in MB per le immagini compresse
fprintf('Dimensione totale in MB delle immagini compresse: %.2f MB\n', total_compressed_size / 1e6);

% Rapporto di Compressione
fprintf("Rapporto di compressione: %.2f %%\n", (1 - (total_compressed_size / total_original_size)) * 100);

% Funzione per calcolare la dimensione di una cartella specifica
function folder_size = calculate_folder_size(folder_path)
    % Inizializza la dimensione della cartella
    folder_size = 0;
    
    % Ottieni le informazioni sui file all'interno della directory specificata
    file_info = dir(folder_path);
    
    % Scorri tutti i file nella directory
    for j = 1:length(file_info)
        % Verifica che l'elemento sia un file (non una directory)
        if ~file_info(j).isdir
            % Somma la dimensione del file
            folder_size = folder_size + file_info(j).bytes;
        end
    end
end
