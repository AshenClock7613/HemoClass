% Caricamento dataset della leucemia, reso come MNIST-like
load('leukemia_mnist_format.mat'); % Contiene trainX, trainY, testX, testY

% Ci assicuriamo che le etichette sono vettoriali
trainY = trainY'; % Trasposizione per garantire che sia un vettore di riga
testY = testY';   % Trasposizione per garantire che sia un vettore di riga

% Aggiustiamo le etichette su 0 o 1 (classificazione binaria)
% Verifichiamo i valori delle etichette
unique_labels = unique(trainY);
if any(unique_labels == 0)
    % Le etichette sono già 0 e 1
else
    trainY = trainY - min(unique_labels); % Impostiamo per iniziare da 0
    testY = testY - min(unique_labels);   % Impostiamo per iniziare da 0
end

% Normalizziamo i dati (convertiamo a double e scaliamo a [0, 1])
trainX = double(trainX') / 255; % Trasponiamo per adattarsi al formato previsto di MATLAB (caratteristiche x esempi)
testX = double(testX') / 255;   % Trasponiamo per adattare il formato (caratteristiche x esempi)

% Controlliamo il class balance
num_class0 = sum(trainY == 0);
num_class1 = sum(trainY == 1);
fprintf('Class 0 samples: %d\n', num_class0);
fprintf('Class 1 samples: %d\n', num_class1);

% iperparametri
input_size = size(trainX, 1);    % Numero di funzioni di ingresso
hidden_sizes = [128, 64];        % Dimensioni degli strati nascosti
output_size = 1;                 % Neurone a uscita singola per la classificazione binaria
epochs = 100;                    % Numero massimo di epoche
tol = 1e-5;                      % Criterio di tolleranza per l'arresto

% Inizializziamo i pesi con l'inizializzazione He (standard per ReLU)
W1 = randn(hidden_sizes(1), input_size) * sqrt(2 / input_size);
W2 = randn(hidden_sizes(2), hidden_sizes(1)) * sqrt(2 / hidden_sizes(1));
W3 = randn(output_size, hidden_sizes(2)) * sqrt(2 / hidden_sizes(2));

% Inizializziamo i bias in modo casuale (distribuzione gaussiana)
b1 = zeros(hidden_sizes(1), 1);  % Inizializzare i bias a zero
b2 = zeros(hidden_sizes(2), 1);
b3 = zeros(output_size, 1);

% funzioni di attivazione
relu = @(x) max(0, x);
relu_derivative = @(x) x > 0;
sigmoid = @(x) 1 ./ (1 + exp(-x));

% Le etichette sono scalari 0 o 1
trainY_vec = trainY; % etichette training
testY_vec = testY;   % etichette testing

% Alleniamo la rete neurale utilizzando il gradiente coniugato non lineare (Fletcher-Reeves)
[W1, W2, W3, b1, b2, b3, cost_history] = conjugate_gradient(trainX, trainY_vec, W1, W2, W3, b1, b2, b3, epochs, tol);

% Plottiamo il costo durante il corso delle epoche
figure;
plot(1:length(cost_history), cost_history, '-o');
xlabel('Epoch');
ylabel('Cost');
title('Cost Over Epochs');

% Testing del modello
Z1_test = W1 * testX + b1;
A1_test = relu(Z1_test);
Z2_test = W2 * A1_test + b2;
A2_test = relu(Z2_test);
Z3_test = W3 * A2_test + b3;
A3_test = sigmoid(Z3_test);

% Prevediamo la classe (soglia a 0,5)
testPredictions = A3_test >= 0.5;
% Calcoliamo la precisione sul set di prova
accuracy = sum(testPredictions == testY_vec) / numel(testY_vec) * 100;
fprintf('Test accuracy: %.2f%%\n', accuracy);
% Plottiamo la matrice di confusione
figure;
plotconfusion(testY_vec, testPredictions);
title('Confusion Matrix');

% === definizioni di funzione ===
function [W1, W2, W3, b1, b2, b3, cost_history] = conjugate_gradient(trainX, trainY_vec, W1, W2, W3, b1, b2, b3, epochs, tol)

    % funzioni di attivazione
    relu = @(x) max(0, x);
    relu_derivative = @(x) x > 0;
    sigmoid = @(x) 1 ./ (1 + exp(-x));

    % Utilizziamo la loss binaria di Cross-Entropy
    cost_function = @(A3, Y) -mean(Y .* log(A3 + eps) + (1 - Y) .* log(1 - A3 + eps));

    % Parametri di ricerca della linea
    rho = 0.5;   % Fattore di riduzione per alfa
    c = 1e-4;    % condizione costante di Armijo
    max_ls_iters = 50;  % Iterazioni massime per la ricerca di riga

    % Initializzazione
    grad_old = [];
    direction = [];
    cost_history = zeros(epochs, 1); % Per memorizzare il costo in ogni epoca

    for epoch = 1:epochs
        % Propagazione in avanti
        Z1 = W1 * trainX + b1;
        A1 = relu(Z1);
        Z2 = W2 * A1 + b2;
        A2 = relu(Z2);
        Z3 = W3 * A2 + b3;
        A3 = sigmoid(Z3); % output finale

        % Calcoliamo il costo (Binary Cross-Entropy Loss)
        cost = cost_function(A3, trainY_vec);

        % Backpropagation per calcolare i gradienti
        m = size(trainX, 2); % numeri degli esempi

        dZ3 = A3 - trainY_vec; % Per la loss di BCE con attivazione sigmoide
        dW3 = (dZ3 * A2') / m;
        db3 = sum(dZ3, 2) / m;

        dA2 = W3' * dZ3; % Backprop attraverso W3
        dZ2 = dA2 .* relu_derivative(Z2); % Applichiamo la derivata ReLU a Z2
        dW2 = (dZ2 * A1') / m;
        db2 = sum(dZ2, 2) / m;

        dA1 = W2' * dZ2; % Backprop attraverso W3
        dZ1 = dA1 .* relu_derivative(Z1); % Applichiamo la derivata ReLU a Z1
        dW1 = (dZ1 * trainX') / m;
        db1 = sum(dZ1, 2) / m;

        % Combiniamo tutti i gradienti
        grad = [dW1(:); dW2(:); dW3(:); db1(:); db2(:); db3(:)];

        % Passo di gradiente coniugato
        if epoch == 1
            direction = -grad; % La direzione iniziale è il gradiente negativo
        else
            beta = (grad' * grad) / (grad_old' * grad_old); % beta di Fletcher-Reeves 
            direction = -grad + beta * direction; % aggiorniamo la direzione
        end

        % Ricerca lineare per trovare l'alfa ottimale
        alpha = 1.0; % Iniziare con una step iniziale
        J_current = cost; % costo corrente
        for ls_iter = 1:max_ls_iters
            % Aggiorniamo temporaneamente i pesi e i bias
            W1_temp = W1 + alpha * reshape(direction(1:numel(W1)), size(W1));
            W2_temp = W2 + alpha * reshape(direction(numel(W1) + 1:numel(W1) + numel(W2)), size(W2));
            W3_temp = W3 + alpha * reshape(direction(numel(W1) + numel(W2) + 1:numel(W1) + numel(W2) + numel(W3)), size(W3));

            b1_temp = b1 + alpha * reshape(direction(numel(W1) + numel(W2) + numel(W3) + 1:numel(W1) + numel(W2) + numel(W3) + numel(b1)), size(b1));
            b2_temp = b2 + alpha * reshape(direction(numel(W1) + numel(W2) + numel(W3) + numel(b1) + 1:numel(W1) + numel(W2) + numel(W3) + numel(b1) + numel(b2)), size(b2));
            b3_temp = b3 + alpha * reshape(direction(numel(W1) + numel(W2) + numel(W3) + numel(b1) + numel(b2) + 1:end), size(b3));

            % Passaggio in avanti con pesi aggiornati
            Z1_temp = W1_temp * trainX + b1_temp;
            A1_temp = relu(Z1_temp);
            Z2_temp = W2_temp * A1_temp + b2_temp;
            A2_temp = relu(Z2_temp);
            Z3_temp = W3_temp * A2_temp + b3_temp;
            A3_temp = sigmoid(Z3_temp);

            % Calcolo del nuovo costo
            J_new = cost_function(A3_temp, trainY_vec);

            % Controllare la condizione di Armijo
            if J_new <= J_current + c * alpha * grad' * direction
                break; % condizione di Armijo soddisfatta, interrompiamo la line ricerca
            else
                alpha = alpha * rho; % Ridurre l'alfa
            end
        end

        % Aggiorniamo i pesi e i bias
        W1 = W1 + alpha * reshape(direction(1:numel(W1)), size(W1));
        W2 = W2 + alpha * reshape(direction(numel(W1) + 1:numel(W1) + numel(W2)), size(W2));
        W3 = W3 + alpha * reshape(direction(numel(W1) + numel(W2) + 1:numel(W1) + numel(W2) + numel(W3)), size(W3));

        b1 = b1 + alpha * reshape(direction(numel(W1) + numel(W2) + numel(W3) + 1:numel(W1) + numel(W2) + numel(W3) + numel(b1)), size(b1));
        b2 = b2 + alpha * reshape(direction(numel(W1) + numel(W2) + numel(W3) + numel(b1) + 1:numel(W1) + numel(W2) + numel(W3) + numel(b1) + numel(b2)), size(b2));
        b3 = b3 + alpha * reshape(direction(numel(W1) + numel(W2) + numel(W3) + numel(b1) + numel(b2) + 1:end), size(b3));

        % Memorizziamo il gradiente corrente per la prossima iterazione
        grad_old = grad;

        % Calcoliamo il costo dopo l'aggiornamento del peso
        cost_history(epoch) = J_new; % Store cost for plotting

        % Condizione di arresto basata sulla norma relativa al gradiente
        if norm(grad) < tol
            fprintf('Converged at epoch %d\n', epoch);
            break;
        end

        % Visualizzazione del costo e della norma di gradiente
        fprintf('Epoch %d: Cost = %.4f, Gradient Norm = %.6f\n', epoch, J_new, norm(grad));
    end
end 