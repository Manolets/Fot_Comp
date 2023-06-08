n1=randn(1000);
mean_n1 = mean2(n1);
std_n1 = std2(n1);

n2=randn(1000);
mean_n2 = mean2(n2);
std_n2 = std2(n2);

mean_n1n2 = mean2(n1 + n2);
std_n1n2 = std2(n1 + n2);

mean_n1n2menos = mean2(n1 - n2);
std_n1n2menos = std2(n1 - n2);

% Primera respuesta = std = sqrt(std1.^2 + std1.^2)

%%

m = 5.0;
std3 = 2.0;

new = (n1-mean_n1)/std_n1;
new = new*std3+m;

meannew = mean2(new);
stdnew = std2(new);

%%
stdsn1new = std2(n1 + new);
stdsn2new = std2(n2 + new);

% Esperamos que sea 2.23
% Sale 2.23
% para n1 + n3 se obtiene 3
% El problema no lo sé

%%
clear
raw = imread("black.pgm");
R=raw(1:2:end,1:2:end);
B = raw(2:2:end,2:2:end);
G1 =  raw(1:2:end,2:2:end);
G2 =  raw(2:2:end,1:2:end);
% Adjuntar el código para extraer los demás canales. 

%%

hist(double(G1(:)),500);
% Parece tener forma normal (figura histogramG1.png

%%

meanR = mean2(R);
meanG1 = mean2(G1);
meanG2 = mean2(G2);
meanB = mean2(B);

stdR = std2(R);
stdG1 = std2(G1);
stdG2 = std2(G2);
stdB = std2(B);


% meanB	127.8329	1×1	double
% meanG1	128.0078	1×1	double
% meanG2	127.9375	1×1	double
% meanR	128.0076	1×1	double
% stdB	2.7528	1×1	double
% stdG1	1.8027	1×1	double
% stdG2	1.8493	1×1	double
% stdR	2.6025	1×1	double


% Su valor es 128

%%




%%
load ruido;
E = zeros(11, 1);
S = zeros(11, 1);
offset = 128;
start = 1/1000;
ende = 1/30;
step = (ende - start) / (10);
T = [start:step:ende];

for i=1:11
    frame = ruido{i};
    E(i) = mean2(frame) - offset;
    S(i) = std2(frame);

end

plot(E, T)
   
% Ecurva.png
%%


semilogx(S, E, "rs")

% SE.png


%%

H=[E.^0 E E.^2];
b = S.^2;
c = H\b;

%%

N = E / 0.1229;

% c	[3.7396;0.1229;0]	3×1	double
% c1 = 3.7396
% c3 = 0
% N = E / 0.1229 =
% 954.7437
% 1.3214e+03
% 1.9529e+03
% 2.7041e+03
% 3.8638e+03
% 5.3700e+03
% 7.5319e+03
% 1.0911e+04
% 1.5275e+04
% 2.1654e+04
% 3.0145e+04

%%

e=(100:4000);
s=sqrt(c(1) + c(2).*e +c(3).*(e.^2));

%%
hold on
semilogx(S, E, "rs");
semilogx(s, e, "b");
hold off
% se2.png

%%
percent = 100.*s./e;
plot(e, percent, "b");

%ruidopercent.png

%%

% e = 800 -> y = 1.34
% e = 3200 -> y = 0.776
% A exposiciones bajas el ruido se dispara pk han entrado menos cantidad de
% fotones por la imagen está formada por más datos "estimados"
% me he colado en vd jeje



im = imread("color.jpg");
im = im2double(im);
imshow(im);
% coordenada punto: (690, 715)

%%
punto = im(715, 690, :);
div = punto./mean(punto);
punto./div; % =  0.8235
%%
div = reshape(div, [1, 3]);
% div =  [1.180952380952381,1.019047619047619,0.800000000000000]

%%
im(:, 751:1500, :) = im(:, 751:1500, :)./div;
imshow(im);
% half.png


%%
im = imread("color.jpg");
im = im2double(im);
punto = im(742, 1033, :); % punto = 1033, 742
div = punto./mean(punto);
tmp = reshape(div, [1, 3]);
% div =	[1.0322,0.7428,1.2251]	1×3	double
im(:, 751:1500, :) = im(:, 751:1500, :)./div;
imshow(im);
% Adquiere tonalidad verde


%%

clear
im = imread("raw.pgm");
im = double(im);
max = max(im(:));
min = 128;
im = (im - min)./max;
im(im < 0) = 0;
%fc_pinta_im(im, "1");
% paso1re.png o paso1.png
% esta pregunta ni idea
%%
imsize = size(im);
R = zeros(imsize);
G = zeros(imsize);
B = zeros(imsize);
R(1:2:end,1:2:end) = im(1:2:end,1:2:end);
B(2:2:end,2:2:end) = im(2:2:end,2:2:end);
G(1:2:end,2:2:end) =  im(1:2:end,2:2:end);
G(2:2:end,1:2:end) =  im(2:2:end,1:2:end);
%fc_pinta_im(cat(3,R,G,B),'Mosaico');
% paso2.png y paso2re.png

%%

%rojo

for j=1:2:imsize(1)-2
    R(j+1, 1:2:imsize(2)) = (R(j, 1:2:imsize(2)) + R(j+2, 1:2:imsize(2)))./2;
end

for j=1:2:imsize(2)-2
    R(:, j+1) = (R(:, j) + R(:, j+2))./2;
end

%%
%azul

for j=2:2:imsize(1)-2
    B(j+1, 2:2:imsize(2)) = (B(j, 2:2:imsize(2)) + B(2+2, 2:2:imsize(2)))./2;
end

for j=2:2:imsize(2)-2
    B(:, j+1) = (B(:, j) + B(:, j+2))./2;
end

%%
%verde
for i=2:1:imsize(1)-1
    for j=2:1:imsize(2)-1
        G(i,j) = (G(i-1, j) + G(i+1, j) + G(i, j-1) + G(i, j+1))./4;
    end
end

%% quitamos bordes
R=R(2:end-1,2:end-1);
B=B(2:end-1, 2:end-1);
G=G(2:end-1, 2:end-1);
%%
fc_pinta_im(cat(3,R,G,B),'3');
% paso3.png

%%
%%
clear max
BW = (0.3.*R+0.5.*G+0.2.*B);
m = max(BW, [], 'all');
BW = BW./m;
U0 = 0.5;
U1 = 0.8;
ok=(BW>=U0 & BW<=U1);
%%
bwsize = size(BW);
percent = 100.*sum(ok, "all")./(bwsize(1).*bwsize(2));
%percent	11.2518	1×1	double
%%
RR=mean2(R(ok));
BB=mean2(B(ok));
GG=mean2(G(ok));
RRGGBB = [RR GG BB];
div = RRGGBB./mean(RRGGBB);
%RRGGBB	[0.3623,0.3357,0.5494]	1×3	double
%div	[0.8713,0.8074,1.3213]	1×3	double
%%
R = R./div(1);
G = G./div(2);
B = B./div(3);

%%
sR = 1.65.*R - 0.61.*G - 0.04.*B;
sG = 0.01.*R + 1.27.*G - 0.28.*B;
sB = 0.01.*R - 0.21.*G + 1.20.*B;

%%
sim = cat(3,sR,sG,sB);
sim(sim < 0 | sim > 1) = 0;
%%
fc_pinta_im(sim, "4");
% paso4.png

%%
sim = sim(:, :, :).^0.4167;
%%
fc_pinta_im(sim, "5");
%paso5.png

%%
simhsv = rgb2hsv(sim);
%%
hist(simhsv(:,:, 3),1000); xlim([-0.05 1.05]);

%%
clear min
mx= max(simhsv(:,:,3), [], "all");
mn = min(simhsv(:,:,3), [], "all");
%simhsv(:,:,3) = (simhsv(:,:,3) - mn)./mx;
%simhsv(simhsv < 0) = 0;
% mx	0.9173	1×1	double
% mn	0.2167	1×1	double
%%
V = simhsv(:,:,3);
vsize = size(V);
V = reshape(V, [vsize(1).*vsize(2), 1]);
%%
hist(V(:),1000); xlim([-0.05 1.05]);
% histo.png

%%

V = retocaP(V, 1.05);
% histV.png

%% retocap2.png
sizesim = size(simhsv);
V = reshape(V, [sizesim(1), sizesim(2)]);
%%
simhsv(:,:,3) = V;

%%
simrgb = hsv2rgb(simhsv);
%fc_pinta_im(simrgb, "6");
%paso6.png

%%

S = simhsv(:,:,2);
S = reshape(S, [sizesim(1).*sizesim(2), 1]);
S = retocaP(S, 1.25);
%histS.png

%%
simhsv(:,:,2 ) = reshape(S, [sizesim(1), sizesim(2)]);
%%
simrgb = hsv2rgb(simhsv);
%fc_pinta_im(simrgb, "7");
%paso7.png

%%

simrgb = simrgb * 255;
simrgb = uint8(simrgb);
imwrite(simrgb, "foto.tif");
% 17 MB
% 2020x2038 x 8 bits 12 MB

%%
imshow(simrgb);

%%
imwrite(simrgb,'foto_99.jpg','Quality',99);
%101 KB

%%
imwrite(simrgb,'foto_90.jpg','Quality',90);
% 493 KB

%%
im = imread('foto2.tif');
imwrite(im,'im_99.jpg','Quality',99);
%2.14 MB
imwrite(im,'im_90.jpg','Quality',90);
%957 KB

%%

function P=retocaP(P, k)
    mn = min(P);
    mx = max(P);
    mn = mn.*k;
    mx = mx./k;

    subplot(211);
    hist(P); 
    xlim([-0.05 1.05]);

    P = (P - mn)./(mx-mn);
    
    P(P < 0) = 0;
    P(P > 1) = 1;


    subplot(212);
    hist(P); 
    xlim([-0.05 1.05]);

end








