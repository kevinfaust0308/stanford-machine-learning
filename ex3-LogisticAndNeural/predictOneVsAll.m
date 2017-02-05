function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% 5000x10 with each column being the probability of being in class 0,9,8,...,1
probability_matrix = sigmoid(X * all_theta');

% i dont know how to do vectorize this completely
% i will loop through each row for now and see which class each row belongs to 
for i = 1:m
  % get the ith row 
  curr_row = probability_matrix(i,:);
  
  % determine the index of the column with the largest value 
  [x,ix] = max(curr_row);
  
  % ix represents the index with the largest value but index and class
  % behave the same in this case 
  % therefore the class of the ith row is ix 
  p(i) = ix;
  
end


% =========================================================================


end
