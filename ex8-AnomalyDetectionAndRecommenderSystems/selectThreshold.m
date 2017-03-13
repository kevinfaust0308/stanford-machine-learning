function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % get binary vector of which rows (data examples) are predicted anomalies
    % with this given epsilon
    pred_anomaly = pval < epsilon;
    
    % see how many were predicted to be as an anomaly correctly
    true_positives = sum((pred_anomaly == 1) & (yval == 1));

    % see how many were predicted to be an anomaly when they weren't
    false_positives = sum((pred_anomaly == 1) & (yval == 0));

    % see how many were predicted to be normal when it is an anomaly 
    false_negatives = sum((pred_anomaly == 0) & (yval == 1));

    % calculate the precision
    precision = true_positives/(true_positives + false_positives);
    
    % calculate the recall
    recall = true_positives/(true_positives + false_negatives);
    
    % compute the F1 score 
    F1 = (2*precision*recall)/(precision+recall);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
