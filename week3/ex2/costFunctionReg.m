function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
s = sigmoid(z);

m_theta = size(theta);

J = sum( (-y .* log(s)) .- ((1 .- y) .* log(1 .- s)) ) / m + (lambda * sum(theta(2:m_theta, 1) .* theta(2:m_theta, 1)) / (2 * m));

grad(1) = sum( (s .- y) .* X(:,1) ) ./ m;

for iter = 2:length(theta)

grad(iter) = sum( (s .- y) .* X(:,iter) ) ./ m + (lambda * theta(iter) / m);

end






% =============================================================

end
