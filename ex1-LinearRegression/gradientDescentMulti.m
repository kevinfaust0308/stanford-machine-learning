function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
% size of X in (rows,columns)
s = size(X);
m = s(1);
n = s(2);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    descent_vals = zeros(n,1);
    
    % this is a real value 
    pred_subtract_real = X*theta - y;
    
    for j = 1:n
      
      % get the jth column in our training set
      jth_col = X(:,j);
      
      % multiple each row in the pred_subtract_real with the value in the jth column 
      % gets a mx1 matrix 
      temp = pred_subtract_real .* jth_col;
      
      % get the sum of that m*1 matrix 
      new_value = sum(temp);
      
      % pop this onto the bottom of descent_vals
      descent_vals(j) = new_value;
      
      
    end
    
    % multiply descent_vals by the learning rate and simultaneously update theta 
     theta = theta - alpha * (1/m) * descent_vals;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
