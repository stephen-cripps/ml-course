function [optC, optSigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


maxF=0;
    for C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        for sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
            
            %Use the training set to get a model
            model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
            
            %Make predictions in order to get Precision and Recall
            prediction = svmPredict(model,Xval);
            
            predictedPos=sum(prediction);
            
            actualPos=sum(yval);
            
            truePos=sum(prediction & yval); 
           
            precision = truePos/predictedPos; 
            recall = truePos/actualPos;
            %Use Preceision and recall to get F score

            F = (2*precision*recall)/(precision+recall);
            %if F>maxF, set optvalues 
            if F>maxF
                optC = C;
                optSigma = sigma; 
                maxF = F;
                
            end

        end
    end
% =========================================================================

end
