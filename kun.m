

function main_Function()
    % 定义要搜索的目录
    directories = {'/home/wtxt/a/data/kun_original0911_matlab'};
    disp(directories);
    % 初始化空的cell数组来保存找到的.mat文件
    global matfiles;
    matfiles = {};
    % 遍历所有目录
    for i = 1:length(directories)
        % 获取当前目录下的所有.mat文件
        files = dir(fullfile(directories{i}, '*blur.mat'));
        
        % 如果在当前目录下找到了.mat文件
        if ~isempty(files)
            % 遍历找到的所有.mat文件
            for j = 1:length(files)
                % 保存.mat文件的完整路径
                matfiles{end+1} = fullfile(directories{i}, files(j).name);
            end
        end
    end
    % folder = '/dataf/Research/Jax-AI-for-science/res/2D_lines_SIM_2x_all'; % Replace with your directory
    % files = dir(fullfile(folder, '*.mat'));
    % savefolder = split(folder,'/');
    % savefolder = savefolder(end-1);
    % global savefolder
    disp(matfiles)
    
    for i = 1:length(matfiles)
        path = matfiles(i);
        % 使用 fileparts 函数分割路径
        [pathStr, name, ext] = fileparts(path);

        % 使用 split 函数分割路径字符串
        % splitPath = split(pathStr, '/');

        % 构建新的路径
        savelocation = fullfile(pathStr,name);
        save_path = [savelocation,'.tif'];
        % real_i = files(i).name;
        % real_i = real_i(1:end-4);
        % disp(real_i);
        % tif_path = fullfile(folder,files(i).name)
        % disp(savefolder)
        % savelocation = fullfile('result', savefolder, real_i);
        disp(savelocation);
        disp(save_path);
        % mkdir(savelocation)
        re = SpeckleMain('config_kun.txt',path{1},savelocation);
    end
end
function [re] = SpeckleMain(config_path,mat_path,savelocation)
    save_path = [savelocation,'.tif'];
    % clearvars; clc; format compact; close all;
    fid = fopen(config_path, 'r');%'config_simula.txt'
    C = textscan(fid, '%s%s', 'Delimiter', ':', 'MultipleDelimsAsOne', true);
    fclose(fid);

    n = length(C{1});
    config = struct();
    for i = 1:n
        fieldname = C{1}{i};
        value = C{2}(i);
        value = value{1};
        config.(fieldname) = value;
    end
    % savelocation = sprintf('./result/%s/',config.dataset);
    trynumb = 1;
    v = 'QD_v1';%'om3_v3';
    Colornum = 512;
    G_colorbar = linspace(0,1,Colornum) .';
    myRmap = horzcat(zeros(size(G_colorbar)), G_colorbar, zeros(size(G_colorbar)));
    %%% Specify sub-image files
    L_total = str2double(config.channel);
    %A = load('/home/wtxt/dataf/Research/Jax-AI-for-science/res/2D_BioSR/CCPs/RawSIMData_gt/Cell_001.mat'); %('Data_Corr_cos7_actin_om3_v3.mat'); 
    A = load(mat_path);
    SubIms = A.imggt; %% 3D imaging matrix.
    bg=str2double(config.background);%450;%180; %450;
    SubIms = SubIms-bg;%-500;
    SubIms = permute(SubIms,[3,2,1]);
    %----------- threshold ----------%
    SubIms = max(SubIms,0); %
    %---------------------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %----------- Medianfilt ----------%
    % for p = 1:size(SubIms,3)
    %     SubIms(:,:,p) = medfilt2(SubIms(:,:,p),[2 2]);
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a=0; a1=a*L_total;
    b=7000; b1=b*L_total;
    clims = [a b];%for mean
    clims1 = [a1 b1];%for sum
    %------------------ Mean instead of sum -------------%
    %print('-f27',['mean' num2str(trynumb) '_try' ],'-dpng')
    %figure (2); imagesc(mean(SubIms,3),clims); axis image; colorbar; axis equal; %colormap(myRmap); 
    %
    cropsize = str2double(config.cropsize);%67;%30; %62:10micron (40*2/2:1px=162.5) %62-int6, 100-int6;outof 4ok, 200 - int2 %104 int4 
    A1 = [800,1000];%[359,225];%y_top = A1(1); %x_left = A1(2);
    % img_crop = zeros(cropsize, cropsize, L_total); 
    img_crop = zeros(cropsize, cropsize, L_total); 
    img1= double(SubIms(:,:,1));
    L = size(SubIms,3);
    disp(L);
    %%% Specify some reconstruction parameters
    N_iter = str2double(config.Niter);%300; % 15 -20 seems good1e-5%1000; %150; % number of iterations to run 700-5e-8; 200-1e-7
    ls_iter = 15;%20; % maximum line search sub-iterations
    c1_step = 5; % larger --> better line search
    int_fac = str2double(config.scale);  % %for 40x2/2 interpolation factor 4: 40.62 interpolation factor 2: 81.25
    descent = 2; % 1: gradient; 2: conjugate gradient
    desdir = 0; % 1: new; 2: old
    gamma_max = 1;
    nr = 0; % 1: mitigate noise
    i_mod = 1; % intensity multiplier
    zero_pad = 1; %zero-padding width
    save_m = 0; %1: save figures
    conj_grad_initial_step = 1e-5;%6e-8;%7
    psf_mod = 1;
    sectioning = 1; % 1: enable sectioning
    sec_multi = 0.1; % amount of substraction from other subimages, use value<0.33
    sec_abs = 0; 
    %%% Specify some experimental parameters
    NA = str2double(config.NA);%0.6;%0.5;%0.8; % numerical aperture
    mag = 40*2;%100*2; % optical magnification
    wavelength = str2double(config.wavelength);%515;%560; % fluorescence WL [nm]
    campix = 2*6500; % CCD real pixel size [nm]
    mag_pix = campix/mag; % magnified effective hardware pixel size [nm]
    mag_pix = str2double(config.mag_pix);
    pixelsize = mag_pix/int_fac; % pixel size [nm] after interpolation
    pdim = length(SubIms);
    SubIms = i_mod*SubIms; % increase photon count for better reconstruction

    % NEW
    crop_dim = cropsize; %62;10um
    % crop_dim = pdim
    y_top = A1(1);
    x_left = A1(2);
    y_bottom = y_top + crop_dim -1; % [Y]
    x_right = x_left + crop_dim -1; % [X]
    m = crop_dim.*int_fac;%display(m);
    CropIms = SubIms(y_top:y_bottom,x_left:x_right,:);
    % CropIms = img_crop;

    % CropIms = CropIms - 20000;% mean(CropIms(:));
    % CropIms = CropIms / 40000; % std(CropIms(:));
    CropIms = (CropIms-min(CropIms(:)))/(max(CropIms(:))-min(CropIms(:)));

    disp(min(CropIms(:)));
    disp(max(CropIms(:)));
    pixelvaluescale = str2double(config.pixelvaluescale);
    CropIms = CropIms*pixelvaluescale;
    size(CropIms)
    output = fullfile(savelocation,'orignal.tif');
    [pathStr, name, ext] = fileparts(save_path);
    output = [pathStr, '/',name, '_original', ext];
    disp(output);
    for i = 1:L
        if i==1
            savetif = squeeze(CropIms(:,:,i));
            size(savetif);
            imwrite(im2uint16(savetif/65536),output);
        else
            imwrite(im2uint16(squeeze(CropIms(:,:,i)/65536)),output,"WriteMode",'append');
        end
    end


    %%% Try sectioning
    CropIms = double(CropIms);
    if sectioning ==1
        CropIms_raw = CropIms;
        SecIms = zeros(size(CropIms,1),size(CropIms,2),size(CropIms,3));
        for ind_sec = 1:L
            SecIms(:,:,ind_sec) = CropIms(:,:,ind_sec)-sec_multi.*(CropIms(:,:,1+mod(ind_sec,L))+...
                CropIms(:,:,1+mod(ind_sec+1,L))+CropIms(:,:,1+mod(ind_sec+2,L)));
        end
        CropIms = SecIms;
        if sec_abs == 0
            CropIms(CropIms<0)=0;
        else
            CropIms = abs(CropIms);
        end
    end


    %%
    %%% Apply a hanning window
    %w2 = hann(m);
    w2 = tukeywin(m,0.5);
    w = w2*w2'; % above 2 line is to apply hanning wind to avoid boundry
    % effect.
    %%%%%%%%%%%%%%%%%%%%%%
    w = fspecial('Gaussian',m,m/2);
    w = w-w(m,m);
    w = w.*(w>0);
    w = w./max(w(:));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %figure; imagesc(w);
    %w = ones(m);%this is no hanning window. Use this one as default.
    %%% Interpolate image area
    CropIms = double(CropIms);
    InterpIms = imresize(CropIms, [m m], 'bicubic');
    InterpIms(InterpIms<0) = 0; % correct for interpolation overshoot
    for ind = 1:L_total
        InterpIms(:,:,ind) = w.*InterpIms(:,:,ind);
    end
    %%% Define PSF
    % ref: micro.magnet.fsu.edu/primer/java/imageformation/rayleighdisks/
    % Fraunhofer diffraction: FT of a circle (Airy disk)
    L1 = psf_mod*(0.61*wavelength/NA)/pixelsize; % pixels within Rayleigh criterion
    u = floor(L1)-2;
    [Xpsf,Ypsf] = meshgrid(-u:u,-u:u);
    R = sqrt(Xpsf.^2 + Ypsf.^2); % polar r-coordinate
    R = R*1.220*pi/L1;
    psf = (2*besselj(1,R)./R).^2; psf(u+1,u+1)=1;
    % psf = imread(config.psfpath);
    psf = psf/sum(psf(:));
    filename = 'psf.tif';
    psfsave = im2uint16(psf);
    [pathStr, name, ext] = fileparts(save_path);
    filename = [pathStr, name, '_psf', ext];
    disp(filename);
    imwrite(psfsave,filename);

    % error("xxxx")
    psf = gpuArray(psf);
    %%% Zero-pad sub-images
    z_edge = zero_pad*2*ceil(L1);
    mzp = m + z_edge;
    M = zeros(mzp,mzp,L); r_x = M;
    M(1+z_edge/2:z_edge/2+m,1+z_edge/2:z_edge/2+m,1:L) = InterpIms;
    %% Prepare recon
    re = gpuArray(zeros(m,m,L));
    X0 = gpuArray(zeros(mzp,mzp,L)); % tensor holding initial guesses
    X = gpuArray(zeros(mzp,mzp,L));
    M = gpuArray(M);
    r_x = gpuArray(r_x);
    %%% Define initial guesses [M - Ip*h]
    %X0 = zeros(mzp,mzp,L); % tensor holding initial guesses
    obj = sum(M,3)/L;
    objcoeff = round(sqrt(max(obj(:))));
    obj = objcoeff*obj/max(obj(:)); mean_obj = mean2(obj);
    X0(:,:,1) = obj; % initial guess at object
    X0(1:z_edge/2,:,1) = mean_obj;
    X0(z_edge/2+m+1:end,:,1) = mean_obj;
    X0(:,1:z_edge/2,1) = mean_obj;
    X0(:,z_edge/2+m+1:end,1) = mean_obj;
    X0(:,:,2:L) = ones(mzp,mzp,L-1).*objcoeff; % SIP guesses
    I0 = ones(mzp,mzp)*objcoeff; % initial guess at average illumination

    %%% Define edge mask for cost function evaluation
    maskc = gpuArray(zeros(mzp,mzp));
    maskc(1+z_edge/2:z_edge/2+m,1+z_edge/2:z_edge/2+m) = 1;
    maskcm = gpuArray(repmat(maskc,1,1,L));

    %%% Initiate iteration parameters
    alpha0 = conj_grad_initial_step;
    gamma  = gpuArray(zeros(1,L));
    gammat = gamma;
    g = gpuArray(zeros(mzp,mzp,L));
    d = g; dp = g;
    gp = g;
    F_x = gpuArray(zeros(N_iter,1));
    alcoef = gpuArray(zeros(N_iter,1));
    gammacoef = gpuArray(zeros(N_iter,L));
    fcost =gpuArray( zeros(N_iter,2));
    c1t = [0.9,0.9,0.2,0.2,0.01,0.01,0.001,0.001,1e-4];
    X = sqrt(X0); % sqrt used to enforce positivity constraint

    %% Iterative reconstruction

    for k = 1:N_iter
        c1 = c1t(min(floor((k-1)/c1_step)+1,length(c1t)));
        X_obj = X(:,:,1).^2;

        %%% Calculate residual error [ref: Mudry 2012 Nat. Phot. {eq: s4}]
        for ind_rx = 1:L-1
            r_x(:,:,ind_rx) = M(:,:,ind_rx) - ...
                fconv2(X_obj.*(X(:,:,ind_rx + 1).^2),psf).*maskc;
        end
        Isum = sum(X(:,:,2:L).^2,3);
        XF = L*I0 - Isum;
        r_x(:,:,L) = M(:,:,L) - fconv2(X_obj.*XF,psf).*maskc;
        
        %%% Define cost function {s12}
        F_2d = sum(abs(r_x(:,:,1:L-1)).^2,3);
        F_2d = F_2d + abs((M(:,:,L) - fconv2(X_obj.*XF,psf)).*maskc).^2;
        F_x(k) = sum(F_2d(:)); % total cost: sum of cost in each pixel

        %%% Define gradients {s14}
        g_obj = 0;
        for ind_gr = 1:L-1
            g_obj = g_obj + ((X(:,:,ind_gr + 1).^2).*X(:,:,1)).*fconv2(r_x(:,:,ind_gr),psf);
        end
        g_obj = g_obj + XF.*X(:,:,1).*fconv2(r_x(:,:,L),psf);
        g(:,:,1) = g_obj;
        for ind_g = 2:L
            g(:,:,ind_g) = ...
                (X_obj.*X(:,:,ind_g)).*fconv2(r_x(:,:,ind_g-1)-r_x(:,:,L),psf);
        end
        if descent == 1 %%% gradient
            d = g;
            f_15 = ' [Grad. descent]';
        elseif descent == 2 %%% conjugate gradient {S13}
            f_15 = ' [Conj. grad. descent]';
            for ind_cg = 1:L
                if k ~= 1
                    gamma(ind_cg) = ...
                        sum(sum(g(:,:,ind_cg).*(g(:,:,ind_cg)-gp(:,:,ind_cg))))...
                        /sum(sum(gp(:,:,ind_cg).^2));
                end
            end
            
            %%% Guarantee descent direction
            if desdir == 1
                gamma(gamma<0) = 0; % (new option, may be better)
                
            elseif desdir == 2
                if min(gamma) < 0
                gamma = gpuArray(zeros(1,L));
                end
            end
            
            if gamma_max == 1
                gamma(gamma>10) = 10;
            end
            
            for ind_d = 1:L
                d(:,:,ind_d) = g(:,:,ind_d)+gamma(ind_d)*dp(:,:,ind_d);
            end
            dp = d; gp = g;
        end
        gammacoef(k,:) = gamma;

        %%% Find alpha
        alpha = alpha0;
        gsum = sum(sum(sum(-d.*g.*maskcm))); % scale to matrix magnitude
        for ls_count = 1:ls_iter
            minimprove = F_x(k) + c1*alpha*gsum;
            Xt = X + alpha*d;
            %%% re-calculate cost function
            F_rc = 0; XFi = L*I0;
            for ind_cf = 1:L-1
                F_rc = F_rc + (abs((M(:,:,ind_cf) - ...
                    fconv2((Xt(:,:,1).^2).*(Xt(:,:,ind_cf + 1).^2),psf)).*maskc).^2);
                XFi = XFi - Xt(:,:,ind_cf+1).^2;
            end
            F_rc = F_rc + (abs((M(:,:,L) - ...
                fconv2((Xt(:,:,1).^2).*XFi,psf)).*maskc).^2);
            f1 = sum(sum(F_rc));
            if f1 <= minimprove
                % disp(ls_iter)
                break
            else
                alpha = alpha/2;
            end
        end
        alcoef(k) = alpha;
        fcost(k,:) = [f1 minimprove]; 

        %%% Update X
        X = X + alpha*d;
    end

    %%% Select reconstructed image
    re = X(1+z_edge/2:z_edge/2+m,1+z_edge/2:z_edge/2+m,1).^2;
    F_x = gather(F_x);
    PctR15 = num2str(round(100*(F_x(end-1)-F_x(end))/F_x(end),2));

    im0 = sum(InterpIms,3)/L;
    im0_N = im0/max(max(im0));
    re_N = re/max(max(re));
    im0 = sum(InterpIms,3)/L;
    im0_N = im0/max(max(im0));
    re_N = re/max(max(re));

    %% Plot Fourier transform

    contrast_f = 1; % 0: none; 1: 0-min;
    cimin = 0.58; cimax = 0.95;
    crmin = 0.58; crmax = 0.95;
    f_i = 8; % interpolation factor
    % fftfig = figure(7);
    % set(fftfig, 'units','normalized','outerposition',[0 0 1 1]);
    im0_Nf = im0_N; re_Nf = re_N;
    f_im0 = log(abs(ifftshift(fft2(im0_Nf))));
    f_re = log(abs(ifftshift(fft2(re_Nf))));
    if contrast_f == 1
        min_im = min(min(f_im0)); max_im = max(max(f_im0)); % find min and max
        min_re = min(min(f_re)); max_re = max(max(f_re));
        zshift_im = abs(min([0 min_im])); % get log scale with 0 as min
        zshift_re = abs(min([0 min_re]));
        f_im0 = (f_im0 + zshift_im);
        f_re = (f_re + zshift_re);
        max_fz_im = max_im + zshift_im; max_fz_re = max_re + zshift_re;
        min_fz_im = min_im + zshift_im; min_fz_re = min_re + zshift_re;
        C_txt_im0 = ',[cimin*max_fz_im cimax*max_fz_im]';
        C_txt_re = ',[crmin*max_fz_re crmax*max_fz_re]';
    else
        C_txt_im0 = '';
        C_txt_re = '';
    end
    a_fac = 1; am = round(a_fac*m); a_width = m*pixelsize; % FOV [nm]
    a_c = (1 + round((m/2)-(am/2))):round((m/2)+(am/2)); % look at low-k features
    f_im0_c = f_im0(a_c,a_c);
    f_re_c = f_re(a_c,a_c);
    f_im0_i = imresize(f_im0_c, f_i, 'bicubic');
    f_re_i = imresize(f_re_c, f_i, 'bicubic');
    sf = size(f_im0_i); [Xm,Ym] = meshgrid(-sf/2:sf/2-1, -sf/2:sf/2-1);
    k0 = 2*pi/(wavelength/2); % [rad/nm]
    fxy = 2*pi*(round(-am/2):round(am/2))/a_width; % [radians per aperture width]
    kxy = fxy/k0; % [k/k0]

    %% 1D intensity comparison plot
    re_gpu = re;
    re = gather(re_gpu);
    deg = 0; % positive: CCW
    col_prof = 143; % [X]
    row_profi = 60; % [Y]
    i_fac = 2;
    im0_rot = imrotate(im0,deg,'nearest','loose');
    re_rot = imrotate(re,deg,'nearest','loose');
    im_line = im0_rot(row_profi,:);
    im_line = interp(im_line,i_fac);
    im_line = im_line/max(im_line);
    re_line = re_rot(row_profi,:);
    re_line = interp(re_line,i_fac);
    re_line = re_line/max(re_line);
    x_i = pixelsize*(1e-3)*(1:length(im_line))/i_fac;
    x_r = pixelsize*(1e-3)*(1:length(re_line))/i_fac;

    %% save re and im0
    
    % save([savelocation ['re_' v num2str(trynumb) '_' num2str(A1(1)) num2str(A1(2)) '_' num2str(cropsize) '_try.mat' ]],'re');
    % save([savelocation ['im0_' v num2str(trynumb) '_' num2str(A1(1)) num2str(A1(2)) '_' num2str(cropsize) '_try.mat']],'im0');
    % save([savelocation ['crop_param_' v num2str(trynumb) '_' num2str(A1(1)) num2str(A1(2)) '_' num2str(cropsize) '_try.mat']],'x_left','y_top','crop_dim','pixelsize');
    % save([savelocation ['recon_param_' v num2str(trynumb) '_' num2str(A1(1)) num2str(A1(2)) '_' num2str(cropsize) '_try.mat']],'bg','N_iter','int_fac','conj_grad_initial_step');

    re_obj = re ./w;
    re_obj = (re_obj-min(re_obj(:)))/(max(re_obj(:))-min(re_obj(:)));
    re_obj = im2uint16(re_obj);
    re_SIPs = X(1+z_edge/2:z_edge/2+m,1+z_edge/2:z_edge/2+m,2:L).^2;
    re_SIPs = (re_SIPs-min(re_SIPs(:)))/(max(re_SIPs(:))-min(re_SIPs(:)));
    re_SIPs = im2uint16(re_SIPs);
    % imwrite(re_obj,sprintf('%s/re_obj.tif',savelocation));
    imwrite(re_obj,save_path);
    [pathStr, name, ext] = fileparts(save_path);
    sip_fn = [pathStr,'/', name, '_p', ext];
    disp(sip_fn);
    % size(re_obj)
    for i = 1:(L-1)
        % sip_fn = fullfile(savelocation, 'SIP.tif');
        if i == 1
            imwrite(re_SIPs(:,:,i), sip_fn);
        else
            imwrite(re_SIPs(:,:,i), sip_fn, 'WriteMode', 'append');
        end
    end
    %close all;
    % toc;
end

