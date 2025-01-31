function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% loop through each example
for i = 1:m
  
  % get data for this row (1xn --> nx1)
  input_layer = X(i,:)';
  
  % get the actual value that this set of features corresponds to 
  real_num = y(i,1);
  
  % change to a binary vector 
  vector_answer = zeros(num_labels,1);
  vector_answer(real_num) = 1;
  
  % perform forward propagation
  % add bias term 
  input_layer = [1; input_layer];
  
  % calculate the second activation 
  activation_second = sigmoid(Theta1 * input_layer);
  
  % add bias term to second activation 
  activation_second = [1; activation_second];
  
  % calculate the output 
  output_layer = sigmoid(Theta2 * activation_second);
  
  % calculate and add the cost for the output
  J = J + sum((-vector_answer .* log(output_layer)) - (1-vector_answer).*log(1-output_layer))/m;
  
end

% add reguarlization to cost
% remove bias column in the theta matrix and square every value 
% and take the total sum 
theta1_reg = sum(sum(Theta1(:,2:end).^2));
theta2_reg = sum(sum(Theta2(:,2:end).^2));
J = J +(lambda/(2*m))*(theta1_reg+theta2_reg);  

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% loop through each example 
for i = 1:m
  
  % get the data (1xn --> nx1)
  input_layer = X(i,:)';
  
  % get the corresponding y data and convert to binary vector
  real_answer = y(i,1);
  vector_answer = zeros(num_labels,1);
  vector_answer(real_answer) = 1;
  
  % perform forward propagation
  % add bias term 
  input_layer = [1; input_layer];
  
  % calculate the second layer and activation
  second_layer = Theta1 * input_layer; 
  activated_second_layer = [1; sigmoid(second_layer)];
  
  % calculate the output layer and output activation 
  output_layer = Theta2 * activated_second_layer;
  activated_output_layer = sigmoid(output_layer);
  
  % compute delta for last layer 
  output_delta = activated_output_layer - vector_answer;
  
  % compute delta for second layer
  second_layer_delta = (Theta2' * output_delta);
  second_layer_delta = second_layer_delta(2:end);
  second_layer_delta = second_layer_delta .* sigmoidGradient(second_layer);
  
  Theta1_grad = Theta1_grad + second_layer_delta * input_layer';
  Theta2_grad = Theta2_grad + output_delta * activated_second_layer';
  
end 

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add regularization to Theta1_grad
Theta1_grad = [Theta1_grad(:,1)/m ((Theta1_grad(:,2:end)/m)+(lambda/m)*Theta1(:,2:end))];

% add regularization to Theta2_grad
Theta2_grad = [Theta2_grad(:,1)/m ((Theta2_grad(:,2:end)/m)+(lambda/m)*Theta2(:,2:end))];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
