

%%

% Crear la máscara gaussiana
sigma = 2;
L = round(2 * sigma);
filter_size = round(4 * sigma) + 1;
Gs = fspecial('gaussian', [filter_size, filter_size], sigma);

% Cargar la imagen y convertirla a double
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;

% Filtrar la imagen con la máscara gaussiana
imS = imfilter(im, Gs, 'symmetric');

% Calcular el detalle eliminado por el filtrado
detail = im - imS;

% Función para visualizar el detalle obtenido

% Visualizar el detalle obtenido
show_detail(detail);

%%

% Leer la imagen y convertirla a double con valores entre 0 y 255
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;

% Aplicar un filtro gaussiano con S=sigma=2 para obtener la imagen filtrada ims
sigma = 2;
L = 2 * sigma;
hsize = round(4*sigma) + 1;
Gs = fspecial('gaussian', hsize, sigma);
ims = imfilter(im, Gs, 'symmetric');

% Aislar el detalle restando ims de la imagen original
detail = im - ims;

% Reforzar el detalle en la imagen usando alpha=1.5 (150%) en la fórmula anterior
alpha = 1.5;
im2 = im + alpha * detail;

% Construir una imagen formada en su mitad superior por la imagen realzada y en su mitad inferior por la original
[h, w, ~] = size(im);
im3 = [im2(1:h/2,:,:); im(1+h/2:h,:,:)];

imshow(im3)

% 0.24, 0.14-0.16, 0.65-0.67 cielo -> 0.001,0.0085
%0.27-0.30, 0.15.0.19, 0.66-0.94 roca -> 0.01,0.08

%%
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;
S=2;
R=20;
f1=filtro_bilat(im,S,R);

%%
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;
S=2;
L = round(S*2); 
R=20;
f2=imbilatfilt(im,R^2,S,'Padding','symmetric','NeighborhoodSize',2*L+1);

%%

diff = f1-f2;

%%
val = max(max(diff)); % 1.8874e-15

%%
alpha = 2.5;
im2 = im + alpha * det;
imshow(im2)

%%
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;
det=(im-f1);
show_detail(det);

%%
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;
S=5;
R=20;
f1=imbilatfilt(im,R^2,S,'Padding','symmetric','NeighborhoodSize',2*L+1);
f2=imbilatfilt(f1,R^2,S,'Padding','symmetric','NeighborhoodSize',2*L+1);
f3=imbilatfilt(f2,R^2,S,'Padding','symmetric','NeighborhoodSize',2*L+1);
f4=imbilatfilt(f3,R^2,S,'Padding','symmetric','NeighborhoodSize',2*L+1);
f5=imbilatfilt(f4,R^2,S,'Padding','symmetric','NeighborhoodSize',2*L+1);

%%
im = imread('FC_LAB4/img1.jpg');
im = double(im)/255;
S=5;
L = 2 * S;
hsize = round(4*S) + 1;
Gs = fspecial('gaussian', hsize, S);
f1 = imfilter(im, Gs, 'symmetric');
f2 = imfilter(f1, Gs, 'symmetric');
f3 = imfilter(f2, Gs, 'symmetric');
f4 = imfilter(f3, Gs, 'symmetric');
f5 = imfilter(f4, Gs, 'symmetric');
%%
imshow(f5);

%%
flash = imread('FC_LAB4/flash.jpg');
no_flash = imread('FC_LAB4/no_flash.jpg');

S = 5;
R = 10;

L=2*S;
H=fspecial('gaussian',2*L+1,S); 
no_flash_gauss=imfilter(no_flash,H,'sym');
%no_flash_gauss = filtro_gauss(no_flash, S);
no_flash_cross = filtro_cross(no_flash,flash,S,R);

imtot = [no_flash no_flash_gauss no_flash_cross ];
imshow(imtot);

%%
im = imread('FC_LAB4/img2.bmp');
imshow(im);
pause;

v = (0:255);
edges = v(1)-0.5:v(end)+0.5;
h = histcounts(im(:), edges);

bar(v, h);
xlabel('Valor del píxel');
ylabel('Número de píxeles');

[max_val, max_idx] = max(h);
most_common_val = v(max_idx);
fprintf('El valor más común en la imagen es %d, que aparece %d veces.\n', most_common_val, max_val);

[min_val, min_idx] = min(h);
rarest_val = v(min_idx);
fprintf('El valor más raro en la imagen es %d, que aparece %d veces.\n', rarest_val, min_val);
pause;

total_pixels = numel(im(:,:,1));
h_norm = h / (total_pixels * 3);
prob_50 = h_norm(v == 50);
fprintf('La probabilidad de encontrar un valor de 50 en alguna de las componentes RGB de un píxel es %.2f %%.\n', prob_50 * 100);

T = cumsum(h_norm);

plot(v, T);
xlabel('Valor del píxel');
ylabel('Transformación T(x)');

%%
% Calcular histograma acumulativo
h = imhist(im);
cdf = cumsum(h);

% Normalizar histograma acumulativo
cdf = cdf / numel(im);

% Calcular función de transformación T(x)
L = 256;

% Aplicar transformación a la imagen original
im_new = uint8(interp1(0:L-1, T, double(im)));

% Multiplicar por 255 para obtener valores en el rango 0-255
im_new = uint8(im_new * 255);

% Concatenar imágenes horizontalmente
im_concat = horzcat(im, im_new);

% Visualizar imágenes
imshow(im_concat);
pause;
% Calcular histograma de la nueva imagen
h_new = histcounts(im_new(:), 32);

% Visualizar histogramas juntos
subplot(121); bar(h); title('Original');
subplot(122); bar(h_new); title('Transformada');

%%
I = im2double(imread('FC_LAB4/img2.bmp'));

% Convertir la imagen a escala de grises
I = rgb2gray(I);

% Obtener la imagen w usando el filtro bilateral
w = f_bilateral(I, 2*5, 0.05);

% Obtener la imagen alfa usando w y el parámetro gamma
alfa = f_alfa(w, 0.05);

% Obtener la imagen resultante usando la imagen I y la imagen alfa
I_out = L_alfa(I, alfa);

% Obtener la relación R
R = I_out./(double(I)+0.001);

% Multiplicar cada plano de color de la imagen original por R
im_final = uint8(zeros(size(im)));
for c = 1:size(im, 3)
    im_final(:, :, c) = uint8(double(im(:, :, c)).*R);
end

% Mostrar la imagen original y la imagen final
imshow(im), title('Imagen original');
figure, imshow(im_final), title('Imagen final');

%%
% Cargar la imagen degradada
G = im2double(imread('FC_LAB4/degradado.png'));

% Cargar la máscara K
K = load('FC_LAB4/K.mat').K;


% Visualizar la máscara K
imagesc(K);
colormap(gray);

% Crear una variante K2 de la máscara original
K2 = fliplr(flipud(K));

% Hipótesis inicial para la imagen F a recuperar
F = G;

% Parámetros del algoritmo
num_iter = 400;
epsilon = 0.001;

% Vector para almacenar las desviaciones estándar
sigma_vec = zeros(num_iter,1);

% Bucle principal del algoritmo
for i=1:num_iter
    % Filtrar la hipótesis actual F con la máscara K
    G2 = imfilter(F,K,'sym');
    
    % Calcular el cociente entre G y G2
    Q = G./(G2+epsilon);
    
    % Filtrar el cociente Q con la máscara K2
    Q = imfilter(Q,K2,'sym');
    
    % Actualizar la hipótesis F
    Fnew = F.*Q;
    
    % Forzar a que los valores de F estén en el intervalo [0,1]
    Fnew(Fnew<0) = 0;
    Fnew(Fnew>1) = 1;
    
    % Calcular la desviación entre la imagen F antes y después de ser actualizada
    dF = Fnew - F;
    
    % Calcular la desviación estándar de dF
    sigma = std2(dF);
    sigma_vec(i) = sigma;
    
    % Actualizar la hipótesis F
    F = Fnew;
end

% Mostrar la evolución de la desviación estándar
semilogy(sigma_vec);
xlabel('Número de iteración');
ylabel('Desviación estándar');

% Mostrar la imagen final resultante
figure;
imshow(F);


%%
function I_out = L_alfa(I, alfa)
    I = double(I);
    I_out = zeros(size(I));
    for c = 1:size(I, 3)
        Ic = I(:, :, c);
        for i = 1:size(I, 1)
            for j = 1:size(I, 2)
                I_out(i, j, c) = interp1([0 1], [0 1], (Ic(i, j)+0.001)/(1+0.001)*exp(alfa(i, j)*Ic(i, j)));
            end
        end
    end
end

function w = f_promedio(I, L)
    w = imfilter(I, fspecial('average', 2*L+1), 'replicate');
end

function w = f_bilateral(I, L, R)
    w = zeros(size(I));
    [X, Y] = meshgrid(-L:L, -L:L);
    G = exp(-(X.^2+Y.^2)/(2*L^2));
    I = double(I);
    for i = 1:size(I, 1)
        for j = 1:size(I, 2)
            I_patch = I(max(i-L,1):min(i+L,size(I,1)), max(j-L,1):min(j+L,size(I,2)), :);
            diff = I_patch-I(i,j,:);
            H = exp(-sum(diff.^2, 3)/(2*R^2));
            W = H.*G((max(i-L,1):min(i+L,size(I,1)))-i+L+1,(max(j-L,1):min(j+L,size(I,2)))-j+L+1);
            w(i,j,:) = sum(sum(W.*I_patch, 1), 2)./sum(sum(W, 1), 2);
        end
    end
end

function alfa = f_alfa(w, G)
    w_ = (w-min(w(:)))/(max(w(:))-min(w(:)));
    alfa = -128 + 256./(1+exp(-G*(w_-0.5)));
end

%%

function res=filtro_bilat(im,S,R)

L = round(S*2); 
Gs = fspecial('gaussian',2*L+1,S); %Creación máscara gaussiana
im=double(im)/255;

[N,M,P]=size(im);

% Añadimos margen de L filas y columnas por cada lado
% repitiendo la 1ª/última fila y la 1ª/última columna
% Equivale a opción symmetric en imfilter
im=[im(:,L:-1:1,:) im im(:,M:-1:M-L+1,:)];
im=[im(L:-1:1,:,:);im;im(N:-1:N-L+1,:,:)];
    
res=im*0; s=(-L:L); 
for k=L+1:L+N
  for j=L+1:L+M               
      vec = im(k+s,j+s,:);
      D = bsxfun(@minus,vec,im(k,j,:))/R;
      D2 = sum(D.^2,3);
      Gr = exp(-0.5*D2);
      G = Gr.*Gs(s+L+1,s+L+1);
      G = G/sum(G(:));
      %vec = vec.*Gs;    
      res(k,j,:) = sum(sum(vec.*repmat(G,[1,1,P])));
  end
end

% Quitamos el margen que le hemos añadido al principio
res = res(L+(1:N),L+(1:M),:);

return
end

%%
function res=filtro_cross(im,im2,S,R)

L = round(S*2); 
Gs = fspecial('gaussian',2*L+1,S); %Creación máscara gaussiana
im= double(im);
im2= double(im2); %Convertir im2 a double
[N,M,P]=size(im);

% Añadimos margen de L filas y columnas por cada lado
% repitiendo la 1ª/última fila y la 1ª/última columna
% Equivale a opción symmetric en imfilter
im=[im(:,L:-1:1,:) im im(:,M:-1:M-L+1,:)];
im=[im(L:-1:1,:,:);im;im(N:-1:N-L+1,:,:)];
im2=[im2(:,L:-1:1,:) im2 im2(:,M:-1:M-L+1,:)];
im2=[im2(L:-1:1,:,:);im2;im2(N:-1:N-L+1,:,:)];
    
res=im*0; s=(-L:L); 
for k=L+1:L+N
  for j=L+1:L+M               
      vec = im(k+s,j+s,:);
      vec2 = im2(k+s,j+s,:); %Extraer vecindades de im y im2
      D = bsxfun(@minus,vec2,im2(k,j,:))/R;
      D2 = sum(D.^2,3);
      Gr = exp(-0.5*D2);
      G = Gr.*Gs(s+L+1,s+L+1);
      G = G/sum(G(:));
      res(k,j,:) = sum(sum(vec.*repmat(G,[1,1,P]))); %Aplicar coeficientes a vec
  end
end

% Quitamos el margen que le hemos añadido al principio
res = res(L+(1:N),L+(1:M),:);

return
end

%%




function show_detail(det)
    % Calcular el valor absoluto del detalle
    abs_det = abs(det);
    % Combinar los tres planos de color
    combined = 0.3*abs_det(:,:,1) + 0.55*abs_det(:,:,2) + 0.15*abs_det(:,:,3);
    % Visualizar la imagen y la escala de colores
    imagesc(combined);
    colorbar('vert');
end

