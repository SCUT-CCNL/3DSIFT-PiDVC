
m_filepath = 'D:\SIFT_EXPERI_LOG\Hole_intersect\special\PiDVC\';
m_filepath = [m_filepath, '624_500-Ori300Margin162_162_100-m12_std8_r8-intersect_1875-split20-rot30-cos-GN_Strategy2_CPU_PiSIFT_N16_Ransac30_3.2_Sub33_2020-03-29-07-01.txt']

%the start line number of POI results in the text file
startLine = 48;
%read the text
[x_pos, y_pos, z_pos,zncc, u, v, w,u0, v0, w0, ux, uy, uz, vx, vy, vz, wx, wy, wz,  iteration, range,candidate_num,final_num,edge,ROI_flag,val,distance,mode,converge, radius] = textread(m_filepath, '%d%d%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%d%d%d%d%d%f%f%f%d%d%f','delimiter',',','headerlines', startLine); 

%plot the results
%plot U-displacement
figure('NumberTitle', 'off', 'Name', 'U-displacement');
scatter3(x_pos, y_pos, z_pos, 75, u, '.');
xlabel('x'); ylabel('y'); zlabel('z');
xlim([150, 475]); ylim([150, 475]); zlim([90,410]);
colorbar; 
hold on;
ax = gca;
ax.BoxStyle = 'full';
box on
set(gcf,'color',[1,1,1])
set(gca, 'FontSize', 18)
set(gca, 'FontName', 'Times New Roman')


%plot zncc
figure('NumberTitle', 'off', 'Name', 'ZNCC');
scatter3(x_pos, y_pos, z_pos, 75, zncc, '.');
xlabel('x'); ylabel('y'); zlabel('z');
xlim([150, 475]); ylim([150, 475]); zlim([90,410]);
colorbar; 
caxis([0 1]);
hold on;
ax = gca;
ax.BoxStyle = 'full';
box on
set(gcf,'color',[1,1,1])
set(gca, 'FontSize', 18)
set(gca, 'FontName', 'Times New Roman')

mean(iteration)