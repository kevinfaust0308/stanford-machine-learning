function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% apply e^-z to every single element in the vector/matrix/scalar 
% add 1 to the newly generated vector/matrix/scalar
% divide 1 by every single element in the vector/matrix/scalar 
g = 1./(1+exp(-z));


% =============================================================

end
