% Directory dove è memorizzato il dataset
baseDir = 'training_data_compresso/';

numFolds = 3;

% Dimensione dell'input della rete
imageSize = [64, 64];

% Variabili temporanee per memorizzare le immagini di un fold
allX = [];
allY = [];

% Loop su ciascun fold
for foldIdx = 0:numFolds-1
    foldPath = fullfile(baseDir, ['fold_', num2str(foldIdx)]);
    
    allFolder = fullfile(foldPath, 'all');
    hemFolder = fullfile(foldPath, 'hem');
    
    % (label 1)
    allFiles = dir(fullfile(allFolder, '*.png'));
    fprintf('Trovate %d immagini all in fold %d\n', length(allFiles), foldIdx);
    for i = 1:length(allFiles)
        img = imread(fullfile(allFolder, allFiles(i).name));
        img = imresize(img, imageSize); % Ridimensiona l'immagine
        img = im2double(img); % Standardizza i valori dell'immagine 
        if size(img, 3) == 3
            img = rgb2gray(img); % Conversione delle immagini in scala di grigi (se non lo sono già)
        end
        allX = [allX; img(:)']; % Appiattisce l'immagine e l'aggiunge alla matrice dei dati
        allY = [allY; 0]; % Etichette per la classe all
    end
    % (label 2)
    hemFiles = dir(fullfile(hemFolder, '*.png'));
    fprintf('Trovate %d immagini hem in fold %d\n', length(hemFiles), foldIdx);
    for i = 1:length(hemFiles)
        img = imread(fullfile(hemFolder, hemFiles(i).name));
        img = imresize(img, imageSize); % Ridimensiona l'immagine
        img = im2double(img); % Standardizza i valori dell'immagine 
        if size(img, 3) == 3
            img = rgb2gray(img); % Conversione delle immagini in scala di grigi (se non lo sono già)
        end
        allX = [allX; img(:)']; % Appiattisce l'immagine e l'aggiunge alla matrice dei dati
        allY = [allY; 1]; % Etichette per la classe hem
    end
    fprintf('Lettura fold %d completata\n', foldIdx);
end

% Riassunto dell'elaborazione
fprintf('Numero totale di immagini elaborate: %d\n', size(allX, 1));

% Split del dataset in 80% training_set e 20% test_set
splitRatio = 0.8;
numTrain = round(splitRatio * size(allX, 1));

% Mescolamento delle istanze
idx = randperm(size(allX, 1));
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);

% Costruzione delle variabili rappresentanti il dataset
trainX = allX(trainIdx, :);
trainY = allY(trainIdx, :);
testX = allX(testIdx, :);
testY = allY(testIdx, :);

% Salva il dataset in formato MNIST-like .mat
save('leukemia_mnist_format.mat', 'trainX', 'trainY', 'testX', 'testY');

fprintf('Dati estrapolati e memorizzati in leukemia_mnist_format.mat\n');
