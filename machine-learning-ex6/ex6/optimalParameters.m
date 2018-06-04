function [optC, optSigma]= optimalParameters(X, y, Xval, yval)

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
end
