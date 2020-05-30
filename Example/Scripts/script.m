
m_filepath_Dir = 'D:\Results\';
m_filepath = [m_filepath_Dir, 'Simulated_deformed_large_inhomogeneous.nii_PiDVC_CPU_PiSIFT_N16_Ransac30_3.2_Sub33_2020-05-30-13-53.txt']

%the start line number of POI results in the text file
startLine = 45;
%read the text
[x_pos, y_pos, z_pos,zncc, u, v, w,u0, v0, w0, ux, uy, uz, vx, vy, vz, wx, wy, wz,  iteration,ROI_flag, converge, strategy,candidate_num,final_num] = textread(m_filepath, '%d%d%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%d%d%d%d%d%d','delimiter',',','headerlines', startLine); 

%plot the results
%plot U-displacement distribution
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

%plot W-displacement distribution
figure('NumberTitle', 'off', 'Name', 'W-displacement');
scatter3(x_pos, y_pos, z_pos, 75, w, '.');
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