function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% so basically, we have 401 features to pass in 
% 1 set of these 401 features will give us either a 1,2,...,10 because those are all 
% our classes (num_labels)
%
% with multiclassifications, what we can do is create our normal theta matrix 
% except we need to do this for all the possible classifications
% instead of our typical 1x401 (a single row vector of thetas) theta matrix,
% we have a 10x401 matrix where each row represents a class and 
% will contain the thetas that, when applied to 401 given features, will accurately
% predict the probability of how well the features fit the current class 
% so we basically do our gradient descent thingy 10 times for each theta 
% to find the accurate thetas needed
% 
% we can loop through all the possible classes and do what is called the one-vs-all
% classification. this means that with the passed in y data, we check which rows 
% equal the class that we are currently looping on. we will get a logical array 
% (values of either 0 or 1) where a 1 will say "yes this row will result in that class"
% and 0 will say "nope". so just like normal logistic regression, we give in the 
% training data and correct predictions and using gradient descent/other methods,
% we will get a vector of thetas that accurately predict the probability of
% how well a given set of features will fit in this class 
% and we basically do this for all possible classifications and we result in a 
% matrix of each row having the thetas to predict classes 


% loop through all possible classifications 
for c = 1:num_labels
  
  % determine which rows are equal to this class 
  new_y = y == c;
  
  % initialize initial optimal thetas for this classification 
  initial_theta = zeros(n + 1, 1);

  % options for fminunc 
  options = optimset('GradObj','on','MaxIter',50);
  
  % get the optimal thetas with lowest cost that will accurately predict 
  % the probability of whether a given set of features fits this classification 
  [theta] = fmincg (@(t)(lrCostFunction(t, X, new_y, lambda)), initial_theta, options);
  
  % 401x1 to 1x401 and put that theta matrix into the all_theta matrix 
  all_theta(c,:) = theta';
  
end 

% by this point all_theta will have a 401 thetas in each row, where each 
% row of thetas will calculate the probability of whether a set of features 
% is a likely candidate for each class 
  
  
  
  








% =========================================================================


end
