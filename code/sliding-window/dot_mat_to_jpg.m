% This is a way to read a .mat image file which is created here for the radar sweep using 
% Rob's new example function that renders a cartesian image of radar as a .mat. This 
% then converts it into a jpg of the same dimensions in width and height. For some
% reason the number of channels becomes 3 instead of staying grayscale as how the .mat file was.
% Look into the 'bit depth' of the funtion uint8 to fix this.
% NB The for loop is a pseudo code

% BEWARE: This is converting intensities into 8 bit and leads to 
% Everuything become intensity zero!!!  NOT GOOD. Its not
% Gonna work Find another way.

load('radar_sweeps_tensor.mat')
tensor_uint8 = uint8(radar_sweeps_tensor);
for i in range(each image of the tensor stack)
	imwrite(tensor_uint8(i),'imguint8.jpg')