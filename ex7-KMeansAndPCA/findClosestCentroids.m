function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

m = size(X,1)

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% loop through each data point
for i = 1:m
  % variable to store the distance between the data point and the centroid 
  curr_distance = Inf;
  % variable to store the centroid with the smallest distance 
  curr_centroid = 0;
  % loop through each centroid 
  for j = 1:K
    % calculate the squared distance between the data point and the centroid 
    new_vec = X(i,:) - centroids(j,:);
    new_distance = sum(new_vec.^2);
    
    % if distance is smaller than what we have right now, update 
    if new_distance < curr_distance
      curr_distance = new_distance;
      curr_centroid = j;
    end
  end
    
  % save the closest centroid index in our array
  idx(i) = curr_centroid;
  
end


% =============================================================

end

