clear
obj=VideoReader('movie.mp4');

X=[540 NaN NaN NaN]'; Y=[370 NaN NaN NaN]';
NF = get(obj, 'NumberOfFrames');
fprintf('Numero de Frames NF=%d\n',NF);


figure(1);  
frame=read(obj,1); im_obj=imshow(frame); hold on; 
pp_obj=plot(X,Y,'yo','MarkerFaceCol','y','MarkerSize',4); 
tt=text(X+20,Y+20,['1';'2';'3';'4']);
t_frame=text(15,20,'Frame# 001','Color','y','FontSize',18);
set(tt,'FontWeight','Bold','Color',[0 0 1]);
hold off

pause
  
for k=1:NF

  % Leer y presentar el siguiente frame  
  frame=read(obj,k); set(im_obj,'Cdata',frame);
  
  
  
 % Bucle actualizando posiciones X(j),Y(j) de las 4 esquinas
  for j=1:4    
    
  end

 
 % Actualizar plot y etiquetas de los puntos sobre la imagen.  
 set(pp_obj,'Xdata',X,'Ydata',Y); 
 for z=1:4, set(tt(z) ,'Position',[X(z)+15 Y(z)+15 0]); end  
 set(t_frame,'String',sprintf('Frame# %03d',k));
 drawnow
 
end


