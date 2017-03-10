function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% error vector 
error_vector = X*theta - y;

% COMPUTING COST

% without regularization
J = sum(error_vector.^2)/(2*m);
% add regularization
J = J + (lambda/(2*m))*sum(theta(2:end).^2);


% COMPUTING GRADIENT
grad = (X' * error_vector)/m;
X' * error_vector;

% add regularization to gradient 
grad = [grad(1) ; (grad(2:end) + (lambda/m)*theta(2:end))];

% =========================================================================

end
