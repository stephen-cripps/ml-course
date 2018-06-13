function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
idx = zeros(size(X,1),1);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


%vectorised version seems to work for me but the grader doesn't like it. 

% THE ISSUE WAS THE NUMBER OF DIMENSIONS WASN'T GENERALISED
% for i = 1:size(X,1)
%     dist = 0; 
%     for j = 1:size(X,2)
%         dist = dist + (X(i,j)-centroids(:,j)).^2
%     end
%     
%     [y,idx(i)]=min(dist);
%     
% end

%Redoing vectorised with generalised dimesnion (and using bsxfun!) 

%difference in first dimension - bsxfun allows element wise functions
% bsxfun(@minus,X(:,1),centroids(:,1)');

%Rearrange dimensions to make 3d
X3 = permute(X,[1,3,2]);
cent3 = permute(centroids,[3,1,2]);

%Perform difference calculation on each feature of X
difference = bsxfun(@minus,X3(:,:,:),cent3(:,:,:));

%Sum the square of the 3rd dimension to get distancec in a 2d array
distances = sum(difference.^2, 3);

%Find minimum distance indexes
[y,idx]=min(distances,[],2);

% =============================================================

end

