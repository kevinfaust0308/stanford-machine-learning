function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add 1's column to matrix 
X = [ones(m,1) X];

% 5000x401 * 401x25 = 5000x25
% each row will contain the nodes in the hidden layer
hidden_layer = sigmoid(X * Theta1');

% add 1's column to hidden layer 
hidden_layer = [ones(m,1) hidden_layer];

% 5000x26 * 26x10 = 5000x10
% each row will contain the probability of being in a certain class 
final_layer = sigmoid(hidden_layer * Theta2');

% not sure how to vectorize this completely so i will use a loop 
% determine what class each row falls in 
for c = 1:m
  % get the ith row 
  curr_row = final_layer(c,:);
  
  [x,ix] = max(curr_row);
  
  % index is the respective class 
  p(c,1) = ix;  
  

end

% =========================================================================


end
