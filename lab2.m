%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%coursework: face recognition with eigenfaces

% need to replace with your own path
addpath C:\Users\cicho\OneDrive\Documents\MATLAB\MLforVDA\Lab2%software;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loading of the images: You need to replace the directory 
Imagestrain = loadImagesInDirectory ( 'C:\Users\cicho\OneDrive\Documents\MATLAB\MLforVDA\Lab2/training-set/23x28/');
[Imagestest, Identity] = loadTestImagesInDirectory ( 'C:\Users\cicho\OneDrive\Documents\MATLAB\MLforVDA\Lab2/testing-set/23x28/');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Computation of the mean, the eigenvalues, amd the eigenfaces stored in the
%facespace:
ImagestrainSizes = size(Imagestrain);
Means = floor(mean(Imagestrain));
CenteredVectors = (Imagestrain - repmat(Means, ImagestrainSizes(1), 1));

CovarianceMatrix = cov(CenteredVectors);

[U, S, V] = svd(CenteredVectors);
Space = V(: , 1 : ImagestrainSizes(1))';
Eigenvalues = diag(S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Display of the mean image:
MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);
 
end
figure;
subplot (1, 1, 1);
imshow(MeanImage);
title('Mean Image');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display of the 20 first eigenfaces : Write your code here

for i = 1:20
    Eigenface = uint8 (zeros(28, 23));
    for k = 0:643
            Eigenface( mod (k,28)+1, floor(k/28)+1 ) = (Space(i,k+1)+0.25)*(644);
    end
     subplot (4,5, i);
     imshow(Eigenface);
     title([ num2str(i),'th Eigenface']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Projection of the two sets of images onto the face space:
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);

Threshold =20;

TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));

%Distances contains for each test image, the distance to every train image.
for i=1:TestSizes(1) % 70
    for j=1: TrainSizes(1) % 200
        Sum=0; %initialize sum
        for k=1: Threshold % 20 eigenfaces
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2); % squared error
        end
     Distances(i,j)=Sum;
    end
end

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Display of first 6 recognition results, image per image:
figure;
x=6;
y=2;
for i=1:6
      Image = uint8(zeros(28, 23));
      for k = 0:643
     Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
      end
   subplot (x,y,2*i-1);
    imshow (Image);
    title('Image tested');
    
    Imagerec = uint8(zeros(28, 23));
      for k = 0:643
     Imagerec( mod (k,28)+1, floor(k/28)+1 ) = Imagestrain ((Indices(i,1)),k+1);
      end
     subplot (x,y,2*i);
imshow (Imagerec);
title(['Image recognised with ', num2str(Threshold), ' eigenfaces:',num2str((Indices(i,1))) ]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extra step:
% number of different identities in Identity matrix
% number of occurence of each identity 
% plot of histogram of identities in Identity matrix
id = unique(Identity);
hist = histc(Identity, id);
bar(hist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%recognition rate compared to the number of test images:
%Write your code here to compute the recognition rate using top 20 eigenfaces.
Threshold = 20; %number of eignefaces
Distances = zeros(TestSizes(1),TrainSizes(1));
for i=1:TestSizes(1)
    for j=1: TrainSizes(1)
        Sum=0;
        for k=1: Threshold
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2); % sum of squared errors
        end
     Distances(i,j)=Sum;
    end
end
Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images = zeros(1,40);% Number of test images of one given person.

for i=1:70
    number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
    [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end
% histogram of number of test images
bar(number_of_test_images);
xlabel('different identities');
ylabel('number of test images');

recognised_person = zeros(1,40);
recognitionrate=zeros(1,5);
id_per_images=zeros(1,5);
i=1;
while (i<70)
    id=Identity(1,i);   
    distmin=Values(id,1);
    indicemin=Indices(id,1);
    while (i<70)&&(Identity(1,i)==id)
        if (Values(i,1)<distmin)
            distmin=Values(i,1);
            indicemin=Indices(i,1);
        end
        i=i+1;
    end
    recognised_person(1,id)=indicemin;
    id_per_images(number_of_test_images(1,id))=id_per_images(number_of_test_images(1,id))+1;
    
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;
    end   
end
% histograms of total identities per test images  
bar(id_per_images);
xlabel('number of test images');
ylabel('number of identities');
% histograms recognised identities per test images 
bar(recognitionrate);
xlabel('number of test images');
ylabel('recognised identities');

for  i=1:5
        recognitionrate(1,i)=recognitionrate(1,i)/id_per_images(1,i);
end
averageRR = mean(recognitionrate(1,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%effect of threshold (i.e. number of eigenfaces):   
averageRR=zeros(1,50);% we use here 50 eigenfaces
for t=1:50
  Threshold =t;  
Distances=zeros(TestSizes(1),TrainSizes(1));

for i=1:TestSizes(1)
    for j=1: TrainSizes(1)
        Sum=0;
        for k=1: Threshold
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end
     Distances(i,j)=Sum;
    end
end

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images = zeros(1,40);% Number of test images of one given person.
for i=1:70
    number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;
    [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end

recognised_person=zeros(1,40);
recognitionrate=zeros(1,5);
id_per_images=zeros(1,5);

i=1;
while (i<70)
    id=Identity(1,i);   
    distmin=Values(id,1);
        indicemin=Indices(id,1);
    while (i<70)&&(Identity(1,i)==id)
        if (Values(i,1)<distmin)
            distmin=Values(i,1);
        indicemin=Indices(i,1);
        end
        i=i+1;
    end
    recognised_person(1,id)=indicemin;
    id_per_images(number_of_test_images(1,id))=id_per_images(number_of_test_images(1,id))+1;
    
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;
    end   
end

    for  i=1:5
        recognitionrate(1,i)=recognitionrate(1,i)/id_per_images(1,i);
    end
averageRR(1,t)=mean(recognitionrate(1,:));
end
figure;
plot(averageRR(1,:));
title('Recognition rate against the number of eigenfaces used');
xlabel('number of eigenfaces used');
ylabel('recognition rate accuracy');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%effect of K: You need to evaluate the effect of K in KNN and plot the recognition rate against K. Use 20 eigenfaces here.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% effect of K-NN using 20 eigenfaces and k = 10

averageRR = zeros(1,10); % k =10
Threshold = 20; %eigenfaces used
  for knn = 1:10
    Distances = zeros(TestSizes(1),TrainSizes(1));
    
    for i=1:TestSizes(1)
        for j=1:TrainSizes(1)
            Sum = 0;
            for k=1:Threshold
                Sum = Sum + ((Locationstrain(j,k)-Locationstest(i,k)).^2);
            end
            Distances(i,j) = Sum;
        end
    end
    
    Values = zeros(TestSizes(1),TrainSizes(1));
    Indices = zeros(TestSizes(1),TrainSizes(1));
    for i=1:70
        [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
    end

    recogntionrate = 0;
    for i=1:70
            if (floor( (Indices(i,knn)- 1)/5)+1)==Identity(1,i) % check if the corresponding k nearest neighbor index matches the identity
                recogntionrate = recogntionrate + 1;
            end
    end
    recognitionrate = 100*recogntionrate/70;
    message = sprintf('K = %02d\t| Recognition rate : %.2f%%', knn, recognitionrate);
    disp(message);
    
    averageRR(1,knn) = recognitionrate;
  end
figure;
plot(averageRR(1,:));
title('Recognition rate against k nearest neighbor')
xlabel('k Nearest Neighbor');
ylabel('Recognition Rate');

