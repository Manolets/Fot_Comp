

function im = crear_hdr(im_path)
    P = 9;
    
    T = [1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4];
    
    img = imread(im_path);
    alto = size(img, 1);
    ancho = size(img, 2);
    
    hdr_data_red = zeros(alto, ancho, P);
    hdr_data_blue = zeros(alto, ancho, P);
    hdr_data_green = zeros(alto, ancho, P);
    for i = 1:P
     
        img = imread(sprintf('belg_%d.jpg', i));
        
    
        img_double = im2double(img) * 255;
        
      
        red_channel = img_double(:, :, 3);
        blue_channel = img_double(:, :, 1);
        green_channel = img_double(:, :, 2);
        
        hdr_data_red(:, :, i) = red_channel;
        hdr_data_blue(:, :, i) = blue_channel;
        hdr_data_green(:, :, i) = green_channel;
    end

    Zdata10000 = extraer_datos(hdr_data_green, 10000);
    g10000 = solve_G(Zdata10000, T);
    log2R_green = get_log2R(hdr_data_green, g10000, T);
    log2R_red = get_log2R(hdr_data_red, g10000, T);
    log2R_blue = get_log2R(hdr_data_blue, g10000, T);
    
    log2R_combined = cat(3, log2R_red, log2R_green, log2R_blue);
    hdr = 2 .^ log2R_combined;
    hdrwrite(hdr, "res.hdr");
    im = tonemap(hdr, 'AdjustSaturation', 3);
end

function log2R = get_log2R(hdr_data, g, T)
    [height, width, ~] = size(hdr_data);
    log2R = zeros(height, width);
    
    M = 256;
    t=(1:M)'/(M+1);
    w=(t.*(1-t)).^2;
    w=w/max(w);


    for y = 1:height
        for x = 1:width
            Z = squeeze(hdr_data(y, x, :));
            Z = round(Z, 0)+1;
            g(Z)
            g_Z = g(Z) - log2(T)';
            weights = w(Z);
            weights = weights / sum(weights); 
            log2R(y, x) = sum( weights.^g_Z);
        end
    end
end


function muestra_HDR(hdr_set,T) 

if nargin==0, load HDR_data, end

P = length(T);


figure(1); %colormap(gray(256));
set(gcf,'Pos',[510 50 1400 700]);
DY=0.99-0.32; DX=0.005; ANCHO=0.18; ALTO=0.32;

fot=zeros(1,9); cont=1;
for k=3:-1:1, 
  DX=0.01;   
  for j=3:-1:1 
  fot(cont)=axes('Position',[DX DY ANCHO ALTO]); cont=cont+1;
  imshow(hdr_set(:,:,3*(k-1)+j)/255); colormap(gray);
  set(gca,'Xtick',[],'Ytick',[],'Xcolor','r','Ycolor','r');
  set(gca,'LineWidth',2);
  DX=DX+ANCHO+0.007;  
  %pause(0.5)
  end
  DY=DY-ALTO-0.01;
end  

DY=0.2; 
%DX=DX+0.02, %DY=0.2;
DX=0.59;
ANCHO=0.4; ALTO=0.50;
ejes=axes('Position',[DX DY ANCHO ALTO]);
pix=log2(T)*NaN;
pl=plot(log2(T),pix,'ko:','LineWidth',2); 
set(gca,'Ylim',[-5 260]);
title('Valor del píxel (gris)');
xlabel('log_2(T) de cada toma');

cmap=jet(8);

r=0;
b=1; idx=1; [j i b]=ginput(1);
while(b~=3)    
  cc=cmap(r+1,:);    
   for k=1:9,
    axes(fot(k));   
    hold on;   
    pp=plot(j,i,'o');
    set(pp,'MarkerEdgeColor',cc,'MarkerFaceColor',cc); 
    hold off
   end
  j=round(j); i=round(i); 
  values = reshape(hdr_set(i,j,:),1,P); 
  axes(ejes);
  hold on; plot(log2(T),(values),'o:','Color',cc,'LineWidth',2);
  %set(pl,'Ydata',mean(values));  
  r=mod(r+1,8);
  [j i b]=ginput(1);
end  


end


function Zdata = extraer_datos(hdr_data, N)
    % Obtener dimensiones de la imagen
    [alto, ancho, P] = size(hdr_data);
    
    % Inicializar matriz Zdata
    Zdata = zeros(P, N);
    
    % Mostrar la imagen de la 5ª toma
    figure;
    imshow(hdr_data(:, :, 5), []);
    hold on;
    
    % Bucle para escoger y extraer datos en N puntos
    for n = 1:N
        % Escoger coordenadas (i, j) aleatorias dentro de la imagen
        i = floor(rand * alto) + 1;
        j = floor(rand * ancho) + 1;
        
        % Extraer los datos de las P tomas en la posición (i, j)
        pixel_values = squeeze(hdr_data(i, j, :));
        
        % Verificar que los valores sean crecientes
        if all(diff(pixel_values) >= 0)
            % Guardar los valores en la columna correspondiente de Zdata
            Zdata(:, n) = pixel_values;
            
            % Dibujar un punto verde en la posición seleccionada
            plot(j, i, 'g.');
        end
    end
    
    hold off;
end

function g = solve_G(Zdata, T)
    N = size(Zdata, 2);
    P = size(Zdata, 1);
    Neq = N * (P - 1) + 255;
    b = zeros(Neq, 1);
    num_nonzeros = 2 * N * (P - 1) + 3 * 254 + 1;
    i = zeros(num_nonzeros, 1);
    j = zeros(num_nonzeros, 1);
    v = zeros(num_nonzeros, 1);
    
    En = log2(T);

    M = 256; 
    t=(1:M)'/(M+1); 
    w=(t.*(1-t)).^2; 
    w=w/max(w);
    
    eq_count = 0; % Contador de ecuaciones
    entry = 1;
    for n = 1:N
        Zn = Zdata(:, n);
        [~, index] = min(abs(Zn - 128)); % Valor más cercano a 128
        Zref = Zn(index);
        if index == 1
            Zn = Zn(2:P);
        end 
        if index == P
            Zn = Zn(1:P-1);
        end
        if index ~= 1 && index ~= P
            Zn = Zn([1:index - 1, index + 1:P]);
        end

        for p = 1:P-1
            eq_count = eq_count + 1;
            
            W = 1;% sqrt(w(Zn(p) + 1)*w(Zref + 1));
            
            i(entry) = eq_count;
            j(entry) = Zn(p) + 1;
            v(entry) = 1 * W;
            
            entry = entry + 1;

            i(entry) = eq_count;
            j(entry) = Zref + 1;
            v(entry) = -1 * W;
            b(p + (n-1)*(P-1)) = (En(p) / En(index)) * W;
            
            entry = entry + 1;

        end
    end

    lambda = 0;
    for z = 1:254
        eq_count = eq_count + 1;
        i(entry) = eq_count;
        j(entry) = z;
        v(entry) = -1 * lambda;
        
        entry = entry + 1;

        i(entry) = eq_count;
        j(entry) = z + 1;
        v(entry) = 2 * lambda;
    
        entry = entry + 1;

        i(entry) = eq_count;
        j(entry) = z + 2;
        v(entry) = -1 * lambda;

        entry = entry + 1;
    end
    
    eq_count = eq_count + 1;
    i(entry) = eq_count;
    j(entry) = 129; % Índice correspondiente a g128
    v(entry) = 1  * lambda;

    H = sparse(i, j, v, Neq, 256);

    disp(size(H)); % Dimensiones de H
    disp(sum(H ~= 0)); % Número de elementos no nulos en cada columna
    plot(sum(H ~= 0)); % Gráfico de la distribución de elementos no nulos

    disp(sum(H(:, 50) ~= 0)); % Número de veces que aparece g50 en las ecuaciones
    disp(sum(H(:, 200) ~= 0)); % Número de veces que aparece g200 en las ecuaciones

    g = H\ b;
end