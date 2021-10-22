%function [pano] = MyPanorama()

%% LOAD IN IMAGES
%steps
%below I will load in the train, custom, and test images into cell arrays,
%where each entry will be a since photo in the RGB channel. 

% Must load images from "../Images/Input/", I have the testing,training, and
% custom in that folder.

% Must return the finished panorama.


Set1_file_numbers = [1,2,3];
Set2_file_numbers = [1,2,3];
Set3_file_numbers = [1,2,3,4,5,6,7,8];

%LOADING TRAIN IMAGES
%SET 1
for i = 1:length(Set1_file_numbers)
  jpgFilename = sprintf('%d.jpg', Set1_file_numbers(i));
  fullFileName = fullfile("..\Images\input\Set1\", jpgFilename);                                                                                                 
  Set1_photos{i} = cast(imread(fullFileName), "double");
end

%SET 2
for i = 1:length(Set2_file_numbers)
  jpgFilename = sprintf('%d.jpg', Set2_file_numbers(i));
  fullFileName = fullfile("..\Images\input\Set2\", jpgFilename);                                                                                                 
  Set2_photos{i} = cast(imread(fullFileName), "double");

end

%SET 3
for i = 1:length(Set3_file_numbers)
  jpgFilename = sprintf('%d.jpg', Set3_file_numbers(i));
  fullFileName = fullfile("..\Images\input\Set3\", jpgFilename);                                                                                                 
  Set3_photos{i} = cast(imread(fullFileName), "double");
end




%%
%PANOS
%Set1
panorama_set1_pt1 = load("..\Images\input\panos\pano_Set1_pt1.mat"); %1 and 2
panorama_set1_pt1 = panorama_set1_pt1.panorama;
panorama_set1_final = load("..\Images\input\panos\pano_Set1_final.mat");%3 with panorama_set1_pt1 as image 2
panorama_set1_final = panorama_set1_final.panorama;

%imshow(cast(panorama_set1_final,'uint8'));

%Set2
panorama_set2_pt1 = load("..\Images\input\panos\pano_Set2_pt1.mat"); %2 and 3
panorama_set2_pt1 = panorama_set2_pt1.panorama;
panorama_set2_final = load("..\Images\input\panos\pano_Set2_final.mat");% with 1 and panorama_set2_pt1 as image 1
panorama_set2_final = panorama_set2_final.panorama;
%imshow(cast(panorama_set2_final,'uint8'));

%Set3
panorama_set3_pt1 = load("..\Images\input\panos\pano_Set3_pt1.mat"); %1 and 2
panorama_set3_pt1 = panorama_set3_pt1.panorama;
panorama_set3_pt2 = load("..\Images\input\panos\pano_Set3_pt2.mat"); %3 and panorama_set3_pt1 with panorama_set3_pt1 as image 2
panorama_set3_pt2 = panorama_set3_pt2.panorama;
panorama_set3_pt3 = load("..\Images\input\panos\pano_Set3_pt3.mat"); %4 and panorama_set3_pt2 with panorama_set3_pt2 as image 1
panorama_set3_pt3 = panorama_set3_pt3.panorama;
panorama_set3_pt3 = imresize(panorama_set3_pt3, [5947/4 4931/4]);

%Set1 Project

%Set2 Project

%Set3 Project


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Cylindrical Projection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_1 = Set1_photos{1};
f = 500;
[Ny, Nx, t] = size(image_1);
x = (1:Nx).';
y = (1:Ny).';
new_image_1_projected = zeros(Ny,Nx,3);

%find x_c and y_c
x_c = round(size(image_1,2)/2);
y_c = round(size(image_1,1)/2);
for i = 1:size(image_1,1)
    for j = 1:size(image_1,2)
        x_prime = round(f * tan((x(j)-x_c)/f) + x_c);
        y_prime = round((y(i)-y_c)/cos((x(j)-x_c)/f) + y_c );
        %disp(y_prime)k
        if x_prime < 1 || x_prime > Nx || y_prime < 1 || y_prime > Ny
            image_1_projected(i,j,:) = zeros(1,1,3);
        else
        new_image_1_projected(i,j,:) = image_1(y_prime, x_prime,:);
        end
    end
end

imshow(cast(new_image_1_projected, "uint8"));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %DETECT YOUR CORNERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    image_1_rgb = cast(image_1_projected, "uint8");
    image_1 = rgb2gray(image_1_rgb);
    image_1_cornerscore = cornermetric(image_1);

    image_2_rgb = cast(image_2_projected, "uint8");    
    image_2 = rgb2gray(image_2_rgb);
    image_2_cornerscore = cornermetric(image_2);

    %code below will plot the corners on the original image of image 1
    figure;
    imagesc(cast(Set1_photos{1},"uint8"));
    title('Figure 1: corners detected by cornermetric for image 1 of Set1');

%     x1 = [];
%     y1 = [];
%     for i = 1:size(image_1_cornerscore,1) %450
%         for j = 1:size(image_1_cornerscore,2) %600
%             if image_1_cornerscore(i,j) > 2e-5
%             x1 = [x1 j];
%             y1 = [y1 i];
%             end
%         end
%     end
%     hold on;
%     plot(x1,y1,'r*','MarkerSize', .8);
%     hold off;
% 
%     x2 = [];
%     y2 = [];
%     for i = 1:size(image_2_cornerscore,1) %450
%         for j = 1:size(image_2_cornerscore,2) %600
%             if image_2_cornerscore(i,j) > 2e-5
%             x2 = [x2 j];
%             y2 = [y2 i];
%             end
%         end
%     end


  %faster plotting
    image_1_cornerscore_harris = detectHarrisFeatures(image_1);
    xh = [];
    yh = [];
    for i = 1:size(image_1_cornerscore_harris.Location,1)
            if image_1_cornerscore_harris.Metric(i) > 6e-14
            xh = [xh round(image_1_cornerscore_harris.Location(i,1))];
            yh = [yh round(image_1_cornerscore_harris.Location(i,2))];
            end
    end
    hold on;
    plot(xh,yh,'r*','MarkerSize', 3);
    hold off;
   

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %ANMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Find the localmaxima for both images
image_1_local_maxima = imregionalmax(image_1_cornerscore); %this return a matrix with 0's or 1's of the local maxima corners
image_2_local_maxima = imregionalmax(image_2_cornerscore);

%find n_strong corners for both images
N_strong_image_1 = sum(image_1_local_maxima(:)==1); %this is the number of 1's that appear in both iamges after imregionalmax
N_strong_image_2 = sum(image_2_local_maxima(:)==1);

%NOW findcoordinates of the new corners for both images
x_1 =[];
y_1 =[];
x_2 =[];
y_2 =[];
for i = 1:size(image_1_local_maxima,1)
    for j = 1:size(image_1_local_maxima,2)
        if(image_1_local_maxima(i,j) == 1)
        x_1 = [x_1 , j];
        y_1 = [y_1 , i];
        end
    end
end
%[y_1 , x_1] = find(image_1_local_maxima);

for i = 1:size(image_2_local_maxima,1)
    for j = 1:size(image_2_local_maxima,2)
        if(image_2_local_maxima(i,j) == 1)
        x_2 = [x_2 , j];
        y_2 = [y_2 , i];
        end
    end
end
%[y_2 ,x_2] = find(image_2_local_maxima);


%APPLY ALOGRITHIM 1 for image 1
ED_1 = 0;
r_i1 = [];
for i =1:N_strong_image_1 %initialize each entry in r_i1 to inf
    r_i1(i) = inf;
end
r_i_x1 = [];
r_i_y1 = [];

for i = 1:N_strong_image_1
    for j = 1:N_strong_image_1
        if image_1_cornerscore(y_1(j), x_1(j)) > image_1_cornerscore(y_1(i), x_1(i)) %comparing every point in cornerscore matrix with the rest
            ED_1 = (x_1(j) - x_1(i))^2 + (y_1(j) - y_1(i))^2;
        end
        if ED_1 < r_i1(i)
            r_i1(i) = ED_1;
            r_i_x1(i) = x_1(j);
            r_i_y1(i) = y_1(j);
        end
    end
end

%APPLY ALOGRITHIM 1 for image 2
ED_2 = 0;
r_i2 = [];
for i =1:N_strong_image_2
    r_i2(i) = inf;
end
r_i_x2 = []; %the two matrices below are the coords of the corresponding ED
r_i_y2 = [];

for i = 1:N_strong_image_2
    for j = 1:N_strong_image_2
        if image_2_cornerscore(y_2(j), x_2(j)) > image_2_cornerscore(y_2(i), x_2(i))
            ED_2 = (x_2(j) - x_2(i))^2 + (y_2(j) - y_2(i))^2;
        end
        if ED_2 < r_i2(i)
            r_i2(i) = ED_2;
            r_i_x2(i) = x_2(j);
            r_i_y2(i) = y_2(j);
        end
    end
end

%now sort r_i for image one, make sure to keep track of the coords corresponding to the r_i1(euclidean distance array). 
%first make a cell array 
for i = 1:length(r_i1)
R_i1{i} = [r_i1(i) ; r_i_x1(i) ; r_i_y1(i)]; %so each cell holds the ED, x coord, y coord for each point
end

[B1,sortingpattern1] = sort(r_i1, 'descend');
for i = 1:length(sortingpattern1)
    R_i1_sorted{i} = R_i1{sortingpattern1(i)};
end

%now pick the N_best points of the sorted points for image 1
N_best = 1500; 
for i = 1:N_best
    R_i1_best_points{i} = R_i1_sorted{i};
end


%sort second image and best points again
for i = 1:length(r_i2)
R_i2{i} = [r_i2(i) ; r_i_x2(i) ; r_i_y2(i)]; %so each cell holds the ED, x coord, y coord for each point
end

[B2,sortingpattern2] = sort(r_i2, 'descend');
for i = 1:length(sortingpattern2)
    R_i2_sorted{i} = R_i2{sortingpattern2(i)};
end

%now pick the N_best points of the sorted points for second image
N_best = 1500; 
for i = 1:N_best
    R_i2_best_points{i} = R_i2_sorted{i};
end

% x_plot = [];
% y_plot = [];
% for i = 1:length(R_i2_best_points)
% x_plot = [x_plot R_i2_best_points{i}(2,1)];
% y_plot = [y_plot R_i2_best_points{i}(3,1)];
% end
% 
% figure
% imagesc(cast(Set1_photos{2},"uint8"));    
% 
% hold on;   
% plot(x_plot, y_plot, 'r*', 'MarkerSize', .8);
% hold off; 



%Now plot the best points on the original image of image 1 of Set1 for the REPORT
x_plot = [];
y_plot = [];
for i = 1:length(R_i1_best_points)
x_plot = [x_plot R_i1_best_points{i}(2,1)];
y_plot = [y_plot R_i1_best_points{i}(3,1)];
end

figure;
imagesc(cast(Set1_photos{1}, "uint8"));    %imagesc instead of imshow?
title('Figure 2: Corners of the building after ANMS, best points for image 1 of Set1');

hold on;   
plot(x_plot, y_plot, 'r*', 'MarkerSize', 2);
hold off; 




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %FEATURE DESCRIPTOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%remember we have R_i1_best_points(cell array) are the 250 best points of image 1 
%                 R_i2_best_points(cell array) are the 250 best points of image 2
%[ED;x;y] in each cell

FDs_1 = get_fds(image_1, R_i1_best_points,1);
FDs_2 = get_fds(image_2, R_i2_best_points,0);

%end Why is MyParnorama.m in a function 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %FEATURE MATCHING 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

square_sums = inf(size(FDs_2, 2));
matched_indicies_im1 = [];
matched_indicies_im2 = [];
for i = 1:length(FDs_1)
    feature_im1 = FDs_1{i};
    % Check that the feature in FDs_1 has not been discarded
    square_sums = inf(size(FDs_2, 2));
    if ~isempty(feature_im1)
        % Find sum of square difference to all FDs in image 2 
        square_sums(i) = 0;
        for k = 1:size(FDs_2, 2)
            feature_im2 = FDs_2{k};
            % Check that the feature in FDs_2 has not been discarded
            if ~isempty(feature_im2)
                %disp(sum((feature_im1 .^ 2) - (feature_im2 .^ 2)));
                square_sums(k) = sum((feature_im1 - feature_im2).^2);
            end
        end
        [square_sums, order] = sort(square_sums);
        ratio = square_sums(1) / square_sums(2);
        %disp(ratio)
        if ratio < 0.77 && ratio ~= 0 %ratio of .77 or .75, maybe .65
            matched_indicies_im1 = vertcat(matched_indicies_im1, [R_i1_best_points{i}(2) R_i1_best_points{i}(3)]);
            matched_indicies_im2 = vertcat(matched_indicies_im2, [R_i2_best_points{order(1)}(2) R_i2_best_points{order(1)}(3)]);
        end
    end
end
figure;
showMatchedFeatures(image_1, image_2, matched_indicies_im1, matched_indicies_im2, 'montage');
title("The result of feature matching(note this has innacurate matches)");

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %RANSAC to estimate Robust Homography
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_max = 0
highest_num_inliers = -inf;
while N_max ~= 80000 %% && highest_num_inliers/size(matched_indicies_im1,1) < .90 

    %first pick 4 random points in BOTH matched indices
    randoms = randperm(size(matched_indicies_im1,1),4);
    %p_i = [matched_indicies_im1(randoms,1), matched_indicies_im1(randoms,2)];  %if needed later
    %p_i_prime = [matched_indicies_im2(randoms,1), matched_indicies_im2(randoms,2)]; %if needed later


    %Compute homography(use the est_homography given)
    H = est_homography(matched_indicies_im1(randoms,1), matched_indicies_im1(randoms,2), matched_indicies_im2(randoms,1), matched_indicies_im2(randoms,2)); 
                                                                                                    %H = est_homography(X,Y,x,y)
                                                                                                    %destination X,Y is image 2 ----- source x,y is image 1
                                          

    %find H_pi by using apply_homography function
    [x_transform,y_transform] = apply_homography(H, matched_indicies_im2(:,1), matched_indicies_im2(:,2));% [X, Y] = apply_homography(H, x, y)
    new_points = [x_transform, y_transform];                                          % Use homogrphay matrix H to compute position (x,y) in the source image to
                                                                                      % the position (X,Y) in the destination image
    %????? maybe this instead                                                                                  
%[x_transform,y_transform] = apply_homography(H, matched_indicies_im2(randoms,1), matched_indicies_im2(randoms,2))% [X, Y] = apply_homography(H, x, y)


    %now compute inliers                
    thresh =.28 * 10^2; %or .27 * 10^3 .28 * 10^3
    %disp(ssd(matched_indicies_im1, new_points));
    store_new_indices = ssd(matched_indicies_im1, new_points) < thresh;%get a logical array showing the indices to keep
    temp_num_inliers = sum(store_new_indices);
    if temp_num_inliers > highest_num_inliers
        highest_num_inliers = temp_num_inliers;
        indexes_to_keep = store_new_indices; %indexes_to_keep& store_new_indices;
    end
N_max = N_max +1;
end

% % %% if you want to show the randoms points only
% % figure
% % showMatchedFeatures(image_1, image_2, p_i, p_i_prime, 'montage');

new_matched_indicies_im1 = [];
new_matched_indicies_im2 = [];
for i = 1:size(matched_indicies_im1,1)
    if indexes_to_keep(i) == 1
        new_matched_indicies_im1 = [new_matched_indicies_im1; matched_indicies_im1(i,:)];
        new_matched_indicies_im2 = [new_matched_indicies_im2; matched_indicies_im2(i,:)];
    end
end
figure;
showMatchedFeatures(image_1, image_2, new_matched_indicies_im1, new_matched_indicies_im2, 'montage');
title("The result of our feature matches after using RANSAC to estimate homography");
%H_final = est_homography(new_matched_indicies_im2(:,1), new_matched_indicies_im2(:,2), new_matched_indicies_im1(:,1), new_matched_indicies_im1(:,2));
%H_final = est_homography(new_matched_indicies_im1(:,1), new_matched_indicies_im1(:,2), new_matched_indicies_im2(:,1), new_matched_indicies_im2(:,2));

%TERMINATE PROGRAM IF NOT ENOUGH MATCHES
% if size(new_matched_indicies_im1,1) < 13

%end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Blending Images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Now we have eliminated our bad matches and now have new_matched_indicies_im1 and new_matched_indicies_im2


% show figures for debugging before warping
%figure;
%imshow(image_1_rgb);
%title("image 1");

%figure;
%imshow(image_2_rgb);
%title("image 2");

% using estimate geometric transform
H_final = estimateGeometricTransform2D(new_matched_indicies_im2, new_matched_indicies_im1, "projective");

% Create the empty panorama by finding the limits of the transform on both
% images (identity transform on image 1, homography on image 2) and using
% that to define the size of the panorama canvas as well as the pixel
% extent or size of each pixel in the canvas' coordinate system

[xlim_im2, ylim_im2] = outputLimits(H_final, [1 size(image_2_rgb, 2)], [1 size(image_2_rgb, 1)]);
[xlim_im1, ylim_im1] = outputLimits(projective2d, [1 size(image_1_rgb, 2)], [1 size(image_1_rgb, 1)]);

% lower bound on minimum is 1
xmin = min([1 xlim_im2 xlim_im2]);
ymin = min([1 ylim_im2 ylim_im1]);

% lower bound on maxiumum is the size of the image
xmax = max([size(image_1_rgb, 2) xlim_im2 xlim_im1]);
ymax = max([size(image_1_rgb, 1) ylim_im2 ylim_im1]);

rows = cast(ymax - ymin, "uint32");
cols = cast(xmax - xmin, "uint32");

% create canvas and panoramaview object for imwarp
panorama = zeros(rows, cols, 3);
panoramaView = imref2d([rows cols], [xmin xmax], [ymin ymax]);

% Initialize our blender to use a binary mask that we specify
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

% Warp image 2 using homography and image 1 using an identity in order to 
% translate it to the correct spot in the new canvas
warped_im2 = imwarp(image_2_rgb, H_final, 'OutputView', panoramaView);
warped_im1 = imwarp(image_1_rgb, projective2d, 'OutputView', panoramaView);

% Make masks for images, go through both warped images and if the value 
% of the pixel is nonzero, put a one in the mask.
mask_im2 = ones(size(warped_im2, 1), size(warped_im2, 2));

for row = 1:size(mask_im2, 1)
    for col = 1:size(mask_im2, 2)
        if sum(warped_im2(row, col, :) == 0) == 3
            mask_im2(row, col, :) = 0;
        end
    end
end

mask_im1 = ones(size(warped_im1, 1), size(warped_im1, 2));
for row = 1:size(mask_im1, 1)
    for col = 1:size(mask_im1, 2)
        if sum(warped_im1(row, col, :) == 0) == 3
            mask_im1(row, col, :) = 0;
        end
    end
end

%figure;
%imshow(mask_im2);
%title("mask_im2 image 2");

%figure;
%imshow(mask_im1);
%title("mask_im1 image 1");

% step through the blender's process, starting with im2.
panorama = step(blender, panorama, cast(warped_im2, "double"), cast(mask_im2, "logical"));
panorama = step(blender, panorama, cast(warped_im1, "double"), cast(mask_im1, "logical"));

figure;
imshow(cast(panorama, "uint8"));
title("panorama");









%% Helper functions
function value = ssd(matched_indicies_im2, new_points)
value  = sum((matched_indicies_im2(:,1)-new_points(:,1)).^2 +  (matched_indicies_im2(:,2)-new_points(:,2)).^2,2); %sum only the rows together
%OR value  = sum((matched_indicies_im2 -new_points).^2,2)
end


function FDs = get_fds(image, points,check)
    bool = 1;%Choose bool from 1 to 250 for plotting purposes. 
    imsize_y = size(image, 1);
    imsize_x = size(image, 2);
    FDs = {};
    for i = 1:length(points)
        x_i = points{i}(2);
        y_i = points{i}(3);

        % if the 40x40 patch around a given points is out of bounds of the
        % image, ignore it
        if ~(y_i - 20 < 1 || y_i + 19 > imsize_y || x_i - 20 < 1 || x_i + 19 > imsize_x)
            % Take 40 x 40 patch around the keypoint
            patch = imcrop(image, [(x_i - 20) (y_i - 20) 39 39]);

            % Apply gaussian blur to patch and resize to an 8x8 image
            filter = fspecial('gaussian', 10); %create filter
            patch_filtered = imfilter(patch, filter); %apply filter
            patch_resized = imresize(patch_filtered, [8,8]);
            
            %FOR PLOTTING VISUALIZATION
            if check == 1
                if bool == 10
                    %disp(bool)
                    figure;
                    imshow(patch);
                    title("40 by 40 patch around a feature point");
                    pause(1);
                    figure
                    imshow(imfilter(patch, filter));
                    title("The patch after gaussian blurring");
                    pause(1);
                end
            end

            % Flatten to 64x1 vector and normalize
            vector = reshape(patch_resized, 64,1);
            vector = cast(vector, "double");
            vector = vector - mean(vector);
            vector = vector / std(vector);
            % Add to feature descriptor list for the corresponding image,
            % maintain the indicies of each feature.
            FDs{i} = vector; 
        end

        bool = bool +1; %for plotting purposes. 
    end
end
% % 
% %     points_image1_source = [matched_indicies_im1(randoms(1),1) matched_indicies_im1(randoms(1),2);
% %                          matched_indicies_im1(randoms(2),1) matched_indicies_im1(randoms(2),2)
% %                          matched_indicies_im1(randoms(3),1) matched_indicies_im1(randoms(3),2)
% %                          matched_indicies_im1(randoms(4),1) matched_indicies_im1(randoms(4),2)];
% %         
% %   %  randoms_2 = randperm(size(matched_indicies_im2,1),4); <--MAYBE THIS
% %     points_image2_dest = [matched_indicies_im2(randoms_1(1),1) matched_indicies_im2(randoms_1(1),2);
% %                          matched_indicies_im2(randoms_1(2),1) matched_indicies_im2(randoms_1(2),2)
% %                          matched_indicies_im2(randoms_1(3),1) matched_indicies_im2(randoms_1(3),2)
% %                          matched_indicies_im2(randoms_1(4),1) matched_indicies_im2(randoms_1(4),2)];
% % 
% %     %find H_pi by using apply_homography function
% %     [x_transform,y_transform] = apply_homography(H, points_image1_source(:,1), points_image1_source(:,2))% [X, Y] = apply_homography(H, x, y)
% %     new_points = [x_transform, y_transform];                                          % Use homogrphay matrix H to compute position (x,y) in the source image to
% %                                      
% % 
