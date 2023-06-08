


function res = fusion(im1, im2, im3)
    

    im1 = im2double(imread(im1));
    im2 = im2double(imread(im2));
    im3 = im2double(imread(im3));
    
    
    N = 5;
    p1 = lap(im1, N);
    p2 = lap(im2, N);
    p3 = lap(im3, N);
    
    
    p_combined = cell(1, N);
    
    p_combined{N} = (p1{N} + p2{N} + p3{N}) / 3;
    
    for k = 1:N-1
        p_combined{k} = zeros(size(p1{k}));
        
        for i = 1:size(p1{k}, 1)
            for j = 1:size(p1{k}, 2)
                detail_norms = [norma(p1, k, i, j), norma(p2, k, i, j),norma(p3, k, i, j)];
                [~, max_idx] = max(detail_norms);
                p_combined{k}(i, j, :) = p1{k}(i, j, :);
                
                if max_idx == 2
                    p_combined{k}(i, j, :) = p2{k}(i, j, :);
                elseif max_idx == 3
                    p_combined{k}(i, j, :) = p3{k}(i, j, :);
                end
            end
        end
    end
    
    HDR = p_combined{N}; 
    for k = N-1:-1:1
        HDR = imresize(HDR, 2);  
        HDR = HDR + p_combined{k};
    end

   v0 = min(HDR(:));
v1 = max(HDR(:));
HDR_rescaled = (HDR - v0) / (v1 - v0);
HSV = rgb2hsv(HDR_rescaled);
HSV(:, :, 3) = adapthisteq(HSV(:, :, 3), 'ClipLimit', 0.01);
HSV(:, :, 2) = HSV(:, :, 2).^0.75;  
res = hsv2rgb(HSV);
end


function p = lap(im, N)
    p = cell(1, N);
    
    for k = 1:N-1
        p{k} = im;
        im = imresize(im, 0.5);
        im2 = imresize(im, 2);
        p{k} = p{k} - im2;
    end
    
    p{N} = im;
end


function res = norma(cell, k, i, j)
    res = norm([cell{k}(i, j, 1) cell{k}(i, j, 2) cell{k}(i, j, 3)]);
end