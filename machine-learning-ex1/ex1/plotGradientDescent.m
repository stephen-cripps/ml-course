function  plotGradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
theta_history = zeros(num_iters, 2);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

temp1= theta(1)-alpha*((1/m)*sum(X*theta-y));
temp2= theta(2)-alpha*((1/m)*sum((X*theta-y).*X(:,2)));

theta(1)=temp1;
theta(2)=temp2;
theta_history(iter,:) = theta;

end

plot(theta_history(:,1),theta_history(:,2))