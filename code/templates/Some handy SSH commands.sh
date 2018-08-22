Some handy SSH commands:

To Transfer a file to JADE:
scp -i ~/.ssh/jade /Users/nihaar/Documents/4yp/data/weights/autoencoder_model_weights.h5 nxs10-ihp05@jade.hartree.stfc.ac.uk:~/

To connect to JADE:
ssh nxs10-ihp05@jade.hartree.stfc.ac.uk -i ~/.ssh/jade