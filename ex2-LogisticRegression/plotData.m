function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% find which rows are positive and which rows are negative
positives = find(y==1);
negatives = find(y==0);

% get the rows that are positive from X from the 1st and 2nd columns
positive_test_1 = X(positives,1);
positive_test_2 = X(positives,2);

% get the rows that are negative from X from the 1st and 2nd columns
negative_test_1 = X(negatives,1);
negative_test_2 = X(negatives,2);

scatter(positive_test_1,positive_test_2,'marker','+');
hold on;
scatter(negative_test_1,negative_test_2,'marker','o');





% =========================================================================



hold off;

end
