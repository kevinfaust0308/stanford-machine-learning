function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% theta is n+1 x 1
% x is m x n+1 

sigmoid_val = sigmoid(X * theta); % m x 1 matrix of sigmoid values 

J = (1/m)*sum((-y .* log(sigmoid_val)) - (1-y) .* log(1-sigmoid_val));


% to update all thetas at once, apply the m x 1 sigmoid matrix to each column in 
% the X matrix (m x n+1)
step1 = (sigmoid_val - y) .* X; 
% now each column in this new matrix will represent values for each theta 

% add up each column in the matrix to get a 1 x n+1 matrix 
step2 = sum(step1);

% divide each theta value by m
step3 = step2/m;

% transpose to get a column matrix
grad = step3';


% =============================================================

end
