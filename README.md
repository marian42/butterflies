# Visualizing butterflies from the Natural History Museum

This repository contains code to process images of butterflies from the [data portal](https://data.nhm.ac.uk/) of the [Natural History Museum in London](https://www.nhm.ac.uk/).
From the processed images, a web-based, interactive, hierarchical [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) plot is created.
The code in this repository can be used to reproduce the visualization or to extract the images and use them for something else.
Pre-trained models for all three neural networks are included.

[Click here for the interactive visualization](https://marian42.de/butterflies/).

[Click here for my blog post that explains the data preparation procedure](https://marian42.de/article/butterflies/).

## Usage

This section explains how to recreate the visualization on your machine.
You need a GPU with at least 8 GB of VRAM and ~650 GB of disk space.
The full dataset contains 716,000 images.
You can use less disk space if you use a subset of the dataset, as explained below.

1. On the NHM data portal, [search for "lepidoptera"](https://data.nhm.ac.uk/dataset/56e711e6-c847-4f99-915a-6894bb5c5dea/resource/05ff2255-c38a-40c9-b657-4ccb55ab2feb?q=lepidoptera&field=associatedMediaCount&view_id=6ba121d1-da26-4ee1-81fa-7da11e68f68e&value=&filters=_has_image%3Atrue).
At this point, you can narrow the search if you want to work with a smaller dataset.
Click the *Download* button and request an email with the CSV files.
The CSV files will be ~1.3 GB for the full dataset.

2. Clone this repository and unpack the files `metadata.csv` and `occurence.csv` from the data portal in the *data* directory.

3. Run `create_metadata.py`.
This will create the file `metadata.csv` in the *data* directory.
The resulting CSV file contains a line for each image that will be used.
You can modify the python script or the resulting CSV file if you want to work with a subset of the dataset.

4. Run `download.py`.
This script will download the original images into the `data/raw` directory.
For the full dataset, this will take ~2 weeks and require 452 GB.
The download speed is limited by the NHM servers, which serve around 1 file per second.
You can stop the script and it will resume where you left off.

5. Optional: Train the classifier U-Net. TODO

6. Run `create_images.py`.
This part removes the backgrounds and creates images with an alpha channel using the U-Net that classifies every pixel as background or foreground.
This will create square PNG images of varying sizes in the `data/images_alpha` directory of just the butterfly for each original image.
The script takes ~24 hours for the full dataset and will use ~160 GB of disk space.
You can stop and resume this script.

7. Optional: Train the rotation network. TODO

8. Run `scale_images.py`.
This creates JPG images with a resolution of 128x128 and a white background for each of the PNG images and stores them in the `data/images_rotated_128` directory.
It also uses the rotation network to bring the butterflies into the default rotation using the rotation network.
The rotations are also saved to `data/rotations_calculated.csv`.
You can stop and resume this script.

9. Optional: Train the autoencoder.
Run `train_autoencoder.py`.
This runs indefinetely, until you stop it.
The longer it trains, the better the result.
You can run `train_autoencoder.py continue` to resume training on the previously trained model.
You can run `test_autoencoder.py` to create example pairs of input and reconstructed images in the `data/test` directory. 
Stop the test script after some images have been created.

10. Run `create_latent_codes.py`.
This calculates latent codes for all images in the dataset using the autoencoder.

11. Run `create_tsne.py`.
This calculates the t-SNE embedding.

12. Run `move_points.py`.
This moves points away from each other that would otherwise overlap in the visualization.

13. Run `create_tiles.py`.
This creates the leaflet map tiles for the visualization.

14. Run `create_json.py`.
This creates JSON files for the metadata that will be displayed in the web app.

15. The files for the web app are in the `server` directory.
You can test the web app by going in to the server directory 
and running `python3 -m http.server`.
Go to the address of the server (i.e. [http://0.0.0.0:8000/](http://0.0.0.0:8000/)) to test the web app.

## License

The images of the butterflies are provided by the [Trustees of the Natural History Museum](https://data.nhm.ac.uk/) under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

The code in this repository is provided under the MIT license.