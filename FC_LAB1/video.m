clear
obj=VideoReader('movie.mp4');

X=[540 644 543 648]'; Y=[370 369 474 473]';
NF = get(obj, 'NumFrames');
fprintf('Numero de Frames NF=%d\n',NF);


figure(1);  
frame=read(obj,1); im_obj=imshow(frame); hold on; 
pp_obj=plot(X,Y,'yo','MarkerFaceCol','y','MarkerSize',4); 
tt=text(X+20,Y+20,['1';'2';'3';'4']);
t_frame=text(15,20,'Frame# 001','Color','y','FontSize',18);
set(tt,'FontWeight','Bold','Color',[0 0 1]);
hold off

pause

R = 20;
r = (-R:R);
dx = ones(length(r), 1)*r; dy=dx';
U = zeros(NF, 4);
V = zeros(NF, 4);
U(1, :) = X;
V(1, :) = Y;
  
for k=1:NF

  % Leer y presentar el siguiente frame  
  frame=read(obj,k); set(im_obj,'Cdata',frame);
  frame = rgb2gray(frame);
  
 % Bucle actualizando posiciones X(j),Y(j) de las 4 esquinas
  for j=1:4    
    x = round(X(j));
    y = round(Y(j));
    frame = im2double(frame);
    subimg = frame(y + r, x + r, :);
    S0 = min(subimg(:));
    d = abs(subimg - S0);
    w = exp(-50*d);
    w = w/sum(w(:));
    xw = w .* (x + dx);
    X(j) = sum(xw(:));
    yw = w .* (y + dy);
    Y(j) = sum(yw(:));

  end

 U(k + 1, :) = X;
 V(k + 1, :) = Y;
 % Actualizar plot y etiquetas de los puntos sobre la imagen.  
 set(pp_obj,'Xdata',X,'Ydata',Y); 
 for z=1:4, set(tt(z) ,'Position',[X(z)+15 Y(z)+15 0]); end  
 set(t_frame,'String',sprintf('Frame# %03d',k));
 if k == 400, pause; end
 drawnow
end

%% 

plot(U);
plot(V);

D = zeros(NF);
for i=1:NF, D(i)=sqrt((U(i,1)-U(i, 2))^2 + (V(i, 1)-V(i, 2))^2); end
plot(D);

dInX = zeros(NF-1);
for i=1:NF, dInX(i)=U(i, 1)-U(i + 1, 1); end
plot(dInX);

