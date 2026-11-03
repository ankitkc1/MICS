# Client manifest

The client manifest is a normal experiment configuration file. It tells the
training program which logical clients exist and which modalities each client
has.

For the current BraTS setting:

    client_id,flair,t1ce,t1,t2
    C1,1,1,1,1
    C2,1,1,1,0

A value of 1 means that the modality is available and 0 means that it is
missing. The complete eight-client setting is stored in
brats2020_clients.csv.

The MRI volumes remain at the existing HPC dataset path. Set MICS_DATA_ROOT to
that path before running an experiment. The manifest does not duplicate the
dataset and does not need to list every image.

If the existing dataset uses separate files or folders for each client, add one
optional column such as split_name or split_file and let the future data loader
resolve it relative to MICS_DATA_ROOT.
