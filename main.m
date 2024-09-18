% Caricamento dataset della leucemia, reso come MNIST-like
load('leukemia_mnist_format.mat'); % Contiene trainX, trainY, testX, testY

% Ci assicuriamo che le etichette sono vettoriali
trainY = trainY'; % Trasposizione per garantire che sia un vettore di riga
testY = testY';   % Trasposizione per garantire che sia un vettore di riga


trainX = double(trainX'); % Trasponiamo per adattare al formato previsto di MATLAB (caratteristiche x esempi)
testX = double(testX');

% Convertiamo le etichette in indici a base 1 per la classificazione
trainY = double(trainY);
testY = double(testY);  

% Convertiamo le etichette in formato one-hot utilizzando ind2vec
trainY_vec = ind2vec(trainY); 
testY_vec = ind2vec(testY);   

%% Definiamo e addestriamo la rete neurale (3 strati nascosti) con il gradiente coniugato di Fletcher-Reeves

hiddenLayerSizes = [128, 64]; % Definiamo 3 strati nascosti con 128, 64
net_cgf = patternnet(hiddenLayerSizes, 'traincgf'); % Creiamo una rete neurale utilizzando Fletcher-Reeves CG

% Rimuovere le funzioni di preprocessing
net_cgf.input.processFcns = {};
net_cgf.output.processFcns = {};

% Utilizziamo tutti i dati per la formazione (nessuna validazione o testing durante il training)
net_cgf.divideFcn = 'dividetrain';

% Impostiamo il numero massimo di epoche (iterazioni di training)
net_cgf.trainParam.epochs = 100; % Numero massimo di epoche

% Disattiviamo early stopping
net_cgf.trainParam.max_fail = 1000;

% Configuriamo la funzione di performance per cross-entropy per le attività di classificazione
net_cgf.performFcn = 'crossentropy';

% Addestriamo la rete neurale con CGF
[net_cgf, tr_cgf] = train(net_cgf, trainX, trainY_vec); % Addestriamo le etichette codificate con one-hot

% Mostriamo la loss delle epoche
figure;
plot(tr_cgf.epoch, tr_cgf.perf, '-o');
xlabel('Epoch');
ylabel('Cross-Entropy Loss');
title('Loss Over Epochs (Fletcher-Reeves CG)');

% Testiamo la rete CGF sul set di test
testOutput_cgf = net_cgf(testX);
testPredictions_cgf = vec2ind(testOutput_cgf); % Convertiamo l'output di rete in indici di classe

% Calcoliamo la precisione per la rete CGF
accuracy_cgf = sum(testPredictions_cgf == testY) / numel(testY) * 100;
fprintf('Test accuracy (CGF Network with 3 hidden layers): %.2f%%\n', accuracy_cgf);

% Matrice di confusione per la rete CGF
figure;
plotconfusion(testY_vec, testOutput_cgf);
title('Confusion Matrix for Neural Network (Fletcher-Reeves CG)');

%% Definiamo e addestriamo la rete neurale (3 strati nascosti) con Classic Gradient Descent

net_gd = patternnet(hiddenLayerSizes, 'traingd'); % Creiamo una rete neurale usando la discesa del gradiente classico

% Rimuovere le funzioni di preprocessing
net_gd.input.processFcns = {};
net_gd.output.processFcns = {};

% Utilizziamo tutti i dati per il train
net_gd.divideFcn = 'dividetrain';

% Impostiamo il numero massimo di epoche (iterazioni di train)
net_gd.trainParam.epochs = 100; % Max number of epochs

% Disattiviamo early stopping
net_gd.trainParam.max_fail = 1000;

% Impostiamo il tasso di apprendimento
net_gd.trainParam.lr = 0.01; % Tasso di apprendimento per discesa in pendenza

% Configurariamo la funzione di performance per cross-entropy per le attività di classificazione
net_gd.performFcn = 'crossentropy';

% Alleniamo la rete neurale con discesa gradiente
[net_gd, tr_gd] = train(net_gd, trainX, trainY_vec); % Addestriamo con etichette codificate con one-hot

% Plottiamo la loss delle epoche
figure;
plot(tr_gd.epoch, tr_gd.perf, '-o');
xlabel('Epoch');
ylabel('Cross-Entropy Loss');
title('Loss Over Epochs (Classic Gradient Descent)');

% Testiamo la rete GD sul set di test
testOutput_gd = net_gd(testX);
testPredictions_gd = vec2ind(testOutput_gd); % Convertiamo l'output di rete in indici di classe

% Calcoliamo la precisione per la rete GD
accuracy_gd = sum(testPredictions_gd == testY) / numel(testY) * 100;
fprintf('Test accuracy (GD Network with 3 hidden layers): %.2f%%\n', accuracy_gd);

% Matrice di confusione per la rete GD
figure;
plotconfusion(testY_vec, testOutput_gd);
title('Confusion Matrix for Neural Network (Gradient Descent)');

%% Confrontiamo i risultati
fprintf('Comparison of Models:\n');
fprintf('Fletcher-Reeves CG Network (3 hidden layers) Accuracy: %.2f%%\n', accuracy_cgf);
fprintf('Gradient Descent Network (3 hidden layers) Accuracy: %.2f%%\n', accuracy_gd);
