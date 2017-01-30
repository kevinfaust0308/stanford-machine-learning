function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% COST
% mx1 matrix of each row containing the probability that each row of features
% in X will be a 1
sigmoided_vals = sigmoid(X * theta);

% cost without regularization term
normal_cost = (1/m) * sum((-y .* log(sigmoided_vals)) - (1-y).*log(1-sigmoided_vals));

% regularization term
regularization_term = (lambda/(2*m))*sum(theta(2:end) .^ 2);

% cost with regularization
J = normal_cost + regularization_term;


% REGULARIZING THETA
pred_minus_real = sigmoided_vals - y; % mx1 matrix
new_matrix = (1/m)*sum(pred_minus_real .* X); % 1 x n+1 matrix where each column will have the new theta
new_matrix = new_matrix'; % n+1 x 1

regularization_matrix = (lambda/m) * theta; % n+1 x 1 matrix where each row will be 
% the extra regularization term for each theta 

% add values in regularization matrix to new_matrix; exclude first row 
grad = [new_matrix(1) ; new_matrix(2:end) + regularization_matrix(2:end)]; 

% =============================================================

end
