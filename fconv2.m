%%%%% FFT-BASED 2D CONVOLUTION
%%% By Joe Ponsetto
% size(fc_out) = size(fc_obj)

function [fc_out] = fconv2(fc_obj,fc_psf)
    obj_s = size(fc_obj,1);
    psf_s = size(fc_psf,1);
    % pad for 'convolution'
    FC_OBJ = fft2(fc_obj,obj_s+psf_s-1,obj_s+psf_s-1);
    FC_PSF = fft2(fc_psf,obj_s+psf_s-1,obj_s+psf_s-1);
    fc_out = real(ifft2(FC_OBJ.*FC_PSF));
    ps = (psf_s-1)/2;
    fc_out = fc_out(ps+1:ps+obj_s,ps+1:ps+obj_s);
    end