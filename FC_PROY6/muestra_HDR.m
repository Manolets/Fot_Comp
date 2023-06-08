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


return

