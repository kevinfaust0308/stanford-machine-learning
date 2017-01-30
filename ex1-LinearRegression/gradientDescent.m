function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % mxn+1 * n+1x1 - mx1 = mx1 matrix 
    % mx1 matrix containing the difference between our predicted value and real value
    pred_subtract_real = X*theta - y;
    
    % generate mxn+1 matrix with each column multiplied the values in pred_subtract_real
    % each column will end up having the value for our theta
    new_matrix = pred_subtract_real .* X;
    
    % calculate the sum of each column 
    new_matrix = sum(new_matrix);
    
    % divide each column by alpha/m
    new_matrix = (alpha/m) * new_matrix;
    
    % transpose this new matrix into column matrix form
    new_matrix = new_matrix';
    
    % subtract the current thetas (n+1x1) with these new values (n+1x1)
    theta = theta - new_matrix;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
