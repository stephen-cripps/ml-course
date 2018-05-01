function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
g = sigmoid(X*theta);
J=sum(-y.*log(g)-(1-y).*log(1-g))/m +(lambda/(2*m))*sum(theta(2:end).^2); 

grad(1) = (X(:,1)'*(g-y))/m;
grad(2:end) = (X(:,2:end)'*(g-y))/m + (lambda/m)*theta(2:end);



% =============================================================

grad = grad(:);

end
